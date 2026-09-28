# Informe final — Clasificación multi-etiqueta de discurso de odio en tuits en español (HS / TR / AG)

**Proyecto**: KAGGLE_TOXIC_2026_PUJ — competencia académica `tweet-toxic-2026-puj` (Kaggle id 167384, organización JarvisLab_PUJ).
**Curso**: Procesamiento del Lenguaje Natural, Maestría en Inteligencia Artificial, Pontificia Universidad Javeriana.
**Fecha de cierre del informe**: 2026-09-24. **Deadline de la competencia**: 2026-10-19 23:00 UTC.
**Documento hermano**: `notebooks/10_final_pipeline.ipynb` (31 celdas: 16 de código, todas ejecutadas, y 15 de narrativa), entregable que concentra el 30 % de la nota.

---

## Resumen ejecutivo

El proyecto aborda la clasificación multi-etiqueta de tres columnas binarias —HS (discurso de odio), TR (dirigido) y AG (agresividad)— sobre 1.731 tuits de entrenamiento y 1.154 de prueba, con la restricción explícita de no usar transformers, LLM ni embeddings preentrenados. La métrica oficial se identificó empíricamente como **F1 macro sobre las 3 columnas** mediante una sonda de submission (plantilla todo-1: LB 0.51494). El modelo final es una **unión TF-IDF word(1,2) + char_wb(3,5)** (27.431 features) con **LogisticRegression balanceada C=2.0** por etiqueta y **umbrales por etiqueta 0.44 / 0.51 / 0.46**, con **LB 0.78306** (mejor de 8 envíos) y OOF local 0.7989. Los resultados negativos son parte central de la evidencia: las 10 variantes de preprocesamiento (mejor +0.0035), el post-proceso jerárquico (−0.0056), los bloques densos (−0.009 a −0.032), los ensembles (mejor +0.0023 anidado), la optimización Optuna (OOF 0.8039 → LB 0.77602), el control de capacidad (gap train-OOF no predice el gap local-LB) y el deep learning desde cero (mejor MLP 0.7949 OOF) no superaron la barrera de adopción pre-registrada. Se documenta un **plateau** del leaderboard: el rango 0.77034–0.78306 equivale a **0.99 errores estándar** (n=1.154). La auditoría final emitió veredicto **GO** con condiciones, hoy satisfechas: pipeline sin fuga, reglas cumplidas y resultados reproducibles bit a bit.

Una investigación posterior (fase v2, 2026-09-25) evaluó 24 líneas de investigación bajo un protocolo de validación corregido (grupos fusionados near-duplicado UNION pseudo-autor y holdout congelado). Su mejor candidato —un meta-modelo de dependencia de etiquetas— mejoró en desarrollo (+0,0081) y en el holdout congelado (+0,0180 pareado), pero **no transfirió al test** (LB 0,78163 frente a 0,78306 de V03); el modelo final entregado sigue siendo **V03** (sección 26).

---

## 1. Definición del problema

La competencia plantea una tarea de clasificación de texto multietiqueta: dado el campo `text` de un tuit o comentario en español, predecir tres etiquetas binarias simultáneas:

- **HS = 1**: el texto contiene discurso de odio.
- **TR = 1**: el discurso está dirigido contra una persona o grupo.
- **AG = 1**: el texto es agresivo.

No se trata de categorías mutuamente excluyentes (no es un problema multiclase): las tres salidas se predicen en paralelo y la métrica oficial (F1 Score del leaderboard privado) se calcula sobre las tres columnas. El enunciado oficial pone el énfasis en «el procesamiento y representación del texto para que los modelos clásicos puedan capturar mejor las señales de toxicidad» (`docs/competition_rules.md`).

La estructura observada de las etiquetas es fuertemente condicionada: en los 1.731 ejemplos de entrenamiento, **TR=1 ocurre únicamente cuando HS=1** (0 violaciones), mientras que AG=1 con HS=0 aparece en solo 26 casos (1,5 %). Esto justificó la decisión D-001 de modelar tres clasificadores binarios acoplados (formulación multi-etiqueta) en lugar de una única clase con seis combinaciones, y de evaluar —y finalmente descartar— un post-proceso jerárquico `HS=0 => TR=0, AG=0`. La estructura condicionada es coherente con el esquema de anotación HatEval (Basile et al., 2019, DOI 10.18653/v1/S19-2007), donde TR y AG solo se anotan sobre textos que ya contienen odio.

El problema tiene tres dificultades específicas del corpus: (i) los textos son cortos (6–306 caracteres, mediana 91); (ii) el corpus está normalizado a ASCII, sin tildes ni eñe, por lo que se pierde parte de la señal morfológica del español; y (iii) la frontera entre «insulto interpersonal en contexto de broma o afiliación» y «discurso de odio» es en buena parte extralingüística, como demostró el análisis de error (sección 20).

## 2. Restricciones de la competencia

Las reglas fueron capturadas verbatim en `docs/competition_rules.md`. Las restricciones que condicionaron todas las decisiones técnicas son:

| Restricción | Origen | Implicación |
|---|---|---|
| Prohibido usar transformers o LLM (BERT, RoBERTa, etc.) | Indicación del profesor (regla adicional) | Se descarta toda la familia de modelos preentrenados, pese a que Plaza-del-Arco et al. (2021, DOI 10.1016/j.eswa.2020.114120) muestran su techo superior en español |
| Prohibido usar embeddings preentrenados | «Todo desde cero» | spaCy se admite solo como anotador lingüístico (POS, lemas), nunca como vector (D-003) |
| Vectorizadores entrenados desde cero | Regla del profesor | TF-IDF, conteos y features inducidas solo con datos de entrenamiento |
| El test no puede usarse para entrenamiento | Reglas de la competencia | El test solo se usa con `transform`/`predict`; jamás para ajustar umbrales o seleccionar modelos |
| Máximo 10 envíos por día; solo cuenta la mejor puntuación | Reglas generales | Se priorizó no sobreajustar el leaderboard (D-014) |
| Formato de submission `id,HS,TR,AG` (1.154 filas) | Plantilla oficial | Validación programática de cada submission contra la plantilla |
| Evaluación: 70 % leaderboard privado + 30 % notebook | Reglas académicas | El esfuerzo final se redirigió a notebook e informe (D-014) |

Restricciones autoimpuestas por integridad académica y metodológica:

- **D-002**: prohibición absoluta de consultar o usar etiquetas o corpus externos (los textos originales de HatEval circulan públicamente; usarlos violaría la validez del resultado).
- **D-007**: cualquier lexicón externo debía registrarse en un catálogo con licencia y finalidad, y mantener siempre una variante autocontenida. Al final no se usó ningún lexicón externo.
- **D-004/D-015**: validación agrupada por near-duplicados y por pseudo-autor.
- **Regla de decisión pre-registrada**: adoptar una mejora solo con ≥ +0.005 absoluto en la métrica primaria y mejora en ≥ 3 de 5 pliegues; en la familia word+char, exigir ≥ +0.010 local antes de gastar una sonda de leaderboard (D-012).

La auditoría final (`reports/auditoria_final.md`) verificó el cumplimiento de cada regla con evidencia ejecutable: búsqueda de `transformers|bert|gpt|llama|...` en todo el código (0 usos reales), ausencia de `.vector`/`.tensor`/`similarity()` de spaCy, vocabulario del artefacto final idéntico a un ajuste solo con train, y 8/8 submissions con formato válido. Veredicto: **APROBADO**.

## 3. Estado del arte

La fase de investigación produjo **45 referencias verificadas** (`reports/estado_arte.md`, `reports/estado_arte.csv`, `references.bib`), con DOI resueltos en Crossref, BibTeX de ACL Anthology, PDFs de ACL/CEUR-WS y comprobaciones puntuales en OpenAlex. Cuatro DOIs fueron validados manualmente contra Crossref (MineriaUNAM, HATERecognizer, SINAI y el artículo de comparación de transformers en español).

Los hallazgos que estructuraron el proyecto:

| Trabajo | Aporte | Uso en el proyecto |
|---|---|---|
| Argota Vega et al. (2019), MineriaUNAM, DOI 10.18653/v1/S19-2079 | Ganó la Subtask A en español de HatEval **sin redes ni embeddings preentrenados**: SVM lineal + char n-gramas, n-gramas de palabras funcionales 2-4, n-gramas de puntuación 2-4, POS, patrones morfológicos y series de palabras agresivas (macro-F1 0.73 vs baseline 0.701) | Techo de referencia de un sistema clásico; justifica la ingeniería de representación y el clasificador lineal |
| Nina-Alcocer (2019), DOI 10.18653/v1/S19-2072 | `TFIDF+STRUCT` mejor F1 (0.780); `TFIDF+EMOTIONS` mejor EMR (0.711 vs 0.629); **combinar tres bloques degradó** el rendimiento | Alerta contra acumular features sin ablación; motivó la aproximación conservadora de un solo bloque de representación |
| Plaza-del-Arco et al. (2019), SINAI, DOI 10.18653/v1/S19-2084 | La votación de ensembles (0.707) apenas supera al SVM lineal (0.701) | Evidencia en contra de esperar mejoras automáticas por ensamblar; H10 se planteó como stacking con OOF |
| Basile et al. (2019), HatEval, DOI 10.18653/v1/S19-2007 | TR y AG no son independientes: se anotan condicionadas a HS | Soporte estructural de D-001 y de la hipótesis H8 |
| Zampieri et al. (2019), OffensEval, DOI 10.18653/v1/N19-1144 | El target es una tarea estructuralmente distinta y más difícil que la detección de ofensa | Modelar TR como cabeza separada |
| ElSherief et al. (2021), Latent Hatred, DOI 10.18653/v1/2021.emnlp-main.29 | El odio implícito es un fenómeno diferenciado y difícil | Base conceptual del patrón de falsos negativos de AG/HS sin insulto léxico |
| Malmasi y Zampieri (2017), DOI 10.26615/978-954-452-049-6_062; Mishra et al. (2018), DOI 10.18653/v1/W18-5101 | N-gramas de caracter para lenguaje abusivo | Justificación de la representación `char_wb` |
| Lipton et al. (2014), DOI 10.1007/978-3-662-44851-9_15; Nan et al. (2012), DOI 10.48550/arXiv.1206.4625 | El umbral óptimo para maximizar F1 no es 0.5 | Base de la familia de experimentos de umbral (TH-01, D-012) |
| Silla y Freitas (2011), DOI 10.1007/s10618-010-0175-9 | Clasificación jerárquica | Referencia para el post-proceso HS=0 => TR=0, AG=0 (H8) |
| Wolpert (1992), DOI 10.1016/S0893-6080(05)80023-1 | Stacked generalization | Diseño de los ensembles con evaluación anidada |
| Wiegand et al. (2019), DOI 10.18653/v1/N19-1060; Davidson et al. (2019), DOI 10.18653/v1/W19-3504; Arango et al. (2019), DOI 10.1145/3331184.3331262 | Sesgo de datasets, variabilidad y fragilidad de la validación en detección de odio | Advertencias incorporadas a las limitaciones y al control de pseudo-autores (D-015) |

El balance de compatibilidad con las restricciones fue 37/45 métodos compatibles de forma directa y 8/45 con advertencia (parcial o prohibido por depender de transformers/embeddings). La regla de decisión pre-registrada (≥ +0.005 y ≥ 3/5 pliegues) y la secuencia de experimentos E01–E10 quedaron documentadas en `reports/hipotesis.md` antes de modelar.

## 4. Dataset

Los tres archivos oficiales se descargaron con la API de Kaggle (`src/data.py:download_competition`) y se auditaron en la Fase 0 (`experiments/00_auditoria_datos.py`, `reports/auditoria_resumen.json`, `reports/data_dictionary.csv`).

| Propiedad | Valor verificado |
|---|---|
| `toxic_train.csv` | 1.731 filas; UTF-8; separador `;`; quoting CSV estándar (el quoting `QUOTE_NONE` rompe 13 filas); 205.417 bytes |
| `toxic_test.csv` | 1.154 filas; **cp1252** (6 bytes `0xBF`); solo `id,text`; 127.006 bytes |
| `submission.csv` | Plantilla `id,HS,TR,AG` separada por coma; ids idénticos y en el mismo orden que el test |
| ids | Rango 20001–24999, únicos, disjuntos train/test; sin asociación con las etiquetas (Spearman p>0,49; chi² p>0,20) |
| `text` | 6–306 caracteres, mediana 91; 100 % ASCII en train (sin tildes ni eñe); URLs `t.co`, menciones `@usuario` y hashtags presentes |
| Nulos | 0 en todos los archivos |

Distribución de etiquetas y combinaciones (train):

| Etiqueta | Positivos | Prevalencia | Desbalanceo (neg:pos) |
|---|---:|---:|---:|
| HS | 756 | 43,67 % | 1,29 |
| TR | 519 | 29,98 % | 2,34 |
| AG | 662 | 38,24 % | 1,62 |

| Combinación (HS, TR, AG) | n | Comentario |
|---|---:|---|
| (0,0,0) | 949 | 54,8 % del corpus |
| (1,1,1) | 491 | 28,4 %; régimen «todo tóxico» |
| (1,0,1) | 145 | odio agresivo no dirigido |
| (1,0,0) | 92 | odio no agresivo |
| (1,1,0) | 28 | dirigido sin agresividad |
| (0,0,1) | 26 | agresividad sin odio |
| (0,1,*) | 0 | **no existe**: TR=1 implica HS=1 |

Estructuras de duplicación y agrupamiento detectadas en la auditoría:

- **4 grupos de near-duplicados** (similitud ≥ 0,85) entre train y test, con 7 filas de train y **10 filas de test** afectadas (`data/processed/grupos_near_duplicados.csv`). El recálculo independiente de la auditoría (TF-IDF char_wb 4-4 + coseno sobre 2.885 filas) encontró 22 pares ≥ 0,85, **0 pares en grupos distintos** a los guardados y 4 pares train-test; los grupos guardados son conservadores (sobre-agrupan, nunca infra-agrupan).
- **0 duplicados exactos train-test**; 5 duplicados exactos dentro de test, que incluyen un grupo de 6 textos idénticos (`#¿nombre?`, ids 22557/20477/24998/20081/22614/24635).
- **17 pseudo-autores** (primera mención `@usuario`) con ≥ 3 tweets: 77 tweets (4,4 % de train). En el control ampliado (D-015) se identificaron 740 pseudo-autores distintos, de los cuales 39 tienen más de un tweet.

El análisis de fuga por `id` y por duplicados no encontró mecanismos de fuga explotables más allá del riesgo acotado de los near-duplicados y los pseudo-autores, que se abordó en la validación (sección 17).

## 5. EDA

El EDA profundo (`reports/eda_resumen.md`; 43 artefactos en `reports/eda/`, 26 figuras) estableció las siguientes señales, todas con tamaño de efecto y corrección de Benjamini-Hochberg por familia:

**Longitud y riqueza léxica.** Los textos con TR=1 son notablemente más cortos (delta de Cliff −0,369) que los TR=0; HS (−0,141) y AG (−0,163) muestran el mismo sentido con efecto pequeño. La deduplicación por near-duplicados no cambió las magnitudes (TR: −0,372). La entropía léxica por clase apenas varía (HS=1: 9,062 bits vs HS=0: 9,261; TR=1: 8,752 vs TR=0: 9,321): el vocabulario tóxico no es más repetitivo en términos globales, pero el léxico concreto sí cambia. El TTR global no se usó como feature.

**Estructura de etiquetas.** Correlación phi: HS-TR 0,743; HS-AG 0,831; TR-AG 0,759. La asociación HS-AG es la más fuerte entre etiquetas distintas y explica por qué el par HS-AG concentra el mayor número de errores conjuntos (sección 20). La estructura de dos regímenes (nada tóxico / todo tóxico) hace que el espacio de etiquetas no sea ruido independiente.

**Fenómenos de superficie.** 47,5 % de los textos contiene al menos una mención, 9,3 % un hashtag y 31,2 % una URL. **0 emojis Unicode** (la normalización ASCII de los organizadores los eliminó; la vía de features de emoji quedó cerrada por construcción), 54 emoticones ASCII y 19 textos con risas en patrón estricto (120 con patrón laxo). El corpus contiene elongaciones (`siiii`), puntuación repetida (`!!!`) y palabras duplicadas adyacentes.

**Contraste con near-duplicados.** Al repetir todos los contrastes conservando una sola fila por grupo, el orden y la magnitud de los efectos se mantuvieron (por ejemplo, V de Cramér de `tiene_insulto` con TR: 0,339 → 0,338). La deduplicación elimina solo 4 filas de train: las señales no provienen de los duplicados.

**Estado inicial de las hipótesis** tras el EDA: H1 con apoyo indirecto (evasión ortográfica y texto sin diacríticos), H2 parcial (léxico agresivo + segunda persona asociados a AG/HS), H3 con reservas (targeting medible, pero el riesgo es que la señal sea «es una respuesta»), H6 parcial (rasgos estilométricos significativos pero pequeños), H8 estructural (0 violaciones de la jerarquía) y H9 indirecta (prevalencias distintas de 0,5). H4, H5, H7 y H10 quedaron sin evaluar en ese momento.

## 6. Análisis lingüístico

El análisis lingüístico se concentró en la caracterización de TR (D-005), cuya definición oficial («dirigido contra una persona o grupo») difiere del esquema HatEval (TR como rango del objetivo: individuo vs grupo). Dentro de HS=1 (n=756: 519 TR=1 y 237 TR=0), las diferencias son grandes y sistemáticas:

| Rasgo (dentro de HS=1) | TR=1 | TR=0 | V de Cramér | q |
|---|---:|---:|---:|---:|
| Mención inicial presente | 64,9 % | 9,7 % | 0,512 | 0,00e+00 |
| Referencia colectiva presente | 10,0 % | 57,0 % | 0,504 | 0,00e+00 |
| Léxico colectivo inducido | — | — | 0,458 | 0,00e+00 |
| Segunda persona | — | — | 0,263 | 0,00e+00 |
| Insulto dirigido | — | — | 0,300 | 0,00e+00 |
| URL presente | — | — | 0,345 | 0,00e+00 |
| Imperativo | — | — | 0,030 | — |

Las dos lecturas de TR quedaron sobre la mesa: a favor de «rango del objetivo» están la concentración de respuestas directas con mención (64,9 % vs 9,7 %) y la referencia colectiva explícita casi siempre en TR=0 (57,0 %); a favor de la definición literal están los 13 casos TR=0 con referencia colectiva y mención inicial (podrían ser «dirigidos a un usuario sobre un grupo») y que solo el 68,7 % de los textos con HS=1 tiene TR=1, cuando la lectura literal «dirigido contra persona o grupo» haría esperar valores cercanos al 100 % salvo por la existencia de un régimen de odio no dirigido (noticia, cita, comentario general). La revisión humana prevista en D-005 (40 ejemplos TR=1 + 40 TR=0 con rasgos anotados en `reports/eda/tr_ejemplos_para_revision.csv`) **no se completó**; el modelado usó la etiqueta tal como aparece en los datos y la ambigüedad se declara como limitación. El error analysis refuerza la lectura del objetivo individual: varios falsos negativos de TR son textos que describen a una persona concreta («el agresor … inmigrante ilegal de origen Nigeriano», «este antisemita dando clase»).

## 7. Textometría

La textometría midió los fenómenos de superficie por documento y los contrastó contra cada etiqueta (chi² con V de Cramér corregida por sesgo, Fisher exacto cuando alguna frecuencia esperada < 5, diferencia de proporciones con IC95 de Newcombe y corrección BH por familia). Resumen de los fenómenos más asociados (conjunto completo, q<0,05):

| Etiqueta | Fenómeno | V de Cramér | Sentido |
|---|---|---:|---|
| HS | `tiene_insulto` | 0,355 | más frecuente con HS=1 |
| HS | `tiene_insulto_dirigido` | 0,173 | más frecuente con HS=1 |
| HS | `tiene_mencion_inicial` | 0,126 | más frecuente con HS=1 |
| HS | `tiene_hashtag` | 0,107 | más frecuente con HS=0 |
| TR | `tiene_insulto` | 0,339 | más frecuente con TR=1 |
| TR | `tiene_mencion_inicial` | 0,325 | más frecuente con TR=1 |
| TR | `tiene_insulto_dirigido` | 0,266 | más frecuente con TR=1 |
| TR | `tiene_colectivo_lexico` | 0,252 | más frecuente con TR=0 |
| AG | `tiene_insulto` | 0,355 | más frecuente con AG=1 |
| AG | `tiene_insulto_dirigido` | 0,195 | más frecuente con AG=1 |
| AG | `tiene_hashtag` | 0,127 | más frecuente con AG=0 |

La información mutua ajustada confirmó el mismo orden. En variables continuas por combinación HS×TR×AG, las más discriminantes fueron `n_insultos` (ε²=0,241), `n_patron_colectivo` (ε²=0,125) y `n_palabras` (ε²=0,118). La textometría también documentó los patrones de repetición (elongaciones, puntuación repetida) que sostienen la hipótesis de evasión ortográfica H1 y que fueron el motivo de incluir n-gramas de carácter. Ninguno de estos rasgos se incorporó como bloque explícito al modelo final: entraron implícitamente vía n-gramas y se usaron como variables de diagnóstico en el análisis de error.

## 8. Análisis léxico

El léxico de agresividad y el inventario de referencias colectivas se indujeron **solo de train** con criterios deterministas y documentados (log-odds de lemas, frecuencia mínima, presencia en varios documentos). El léxico de insultos quedó en **150 lemas**, de los cuales 19 forman el núcleo más frecuente. El inventario colectivo incluye términos como `inmigrante`, `ilegal`, `refugiado`.

Hallazgo metodológico central (revisión `ml-reviewer` de la Fase C): estos léxicos inducidos con las etiquetas de train están **sesgados in-sample**; toda medición de su aporte debía inducirlos **dentro de cada pliegue** y reportar el efecto out-of-fold. Las rebanadas definidas con el léxico inducido in-sample (F1 0,368 en `tiene_insulto_lexico=0`) se excluyeron del ranking de debilidad por ser artefactos de construcción.

El efecto de los léxicos inducidos in-fold se midió en la ablación de features densas (sección 19): funcionan como señal (el léxico AG inducido captura 150 lemas; el colectivo captura inmigrante/ilegal/refugiado), pero **concatenarlos al TF-IDF degradó el F1** (−0,009 a −0,032). El detalle de los lexicones y su contrato de inducción quedó en `reports/features_densas_resultados.json`.

## 9. N-gramas

La comparación de representaciones (Fase 4-5, 25 configuraciones, mismo split agrupado y mismo clasificador LR balanceada C=1.0 en las partes A y B) es el resultado experimental más limpio del proyecto:

| ID | Representación | Dimensiones | F1 OOF (0,5) | Δ vs B1 | Pliegues |
|---|---|---:|---:|---:|---:|
| B1 / PRE-00 | TF-IDF word (1,2), min_df=2, sublinear | 4.024 | 0,7587 | — | — |
| REP-03 | TF-IDF word (1,1) | 1.953 | 0,7417 | −0,0170 | 0/5 |
| REP-01 | TF-IDF char_wb (3,5) | 23.407 | 0,7567 | −0,0021 | 2/5 |
| REP-02 | TF-IDF char_wb (2,5) | 25.036 | 0,7511 | −0,0076 | 2/5 |
| REP-04 | TF-IDF word (1,3) | 4.795 | 0,7562 | −0,0025 | 2/5 |
| **REP-05** | **union word (1,2) + char_wb (3,5)** | **27.431** | **0,7865** | **+0,0278** | **5/5** |
| REP-06 | union word (1,2) + char_wb (3,6) | 34.991 | 0,7855 | +0,0268 | 5/5 |

Lectura: **los n-gramas de carácter por sí solos no superan a los de palabra** (H1 no se cumple en su forma fuerte), pero **la unión sí aporta de forma masiva** (H4 apoyada): +0,0278 con mejora en 5/5 pliegues. El carácter añade robustez ante la evasión ortográfica, las elongaciones y la ausencia de diacríticos, mientras que la palabra conserva el léxico temático. La configuración ganadora definitiva (MOD-03b_BEST) sube C a 2.0 y la unión pasa a 0,7927 (+0,0339 vs B1, 5/5 pliegues).

## 10. POS y sintaxis

spaCy `es_core_news_md` se usó exclusivamente como **anotador lingüístico** (POS, lemas, morfología, dependencias), con `exclude=["ner"]` y prohibición explícita de `doc.vector`, `token.vector`, `doc.tensor` y `similarity()` en cualquier feature (D-003), cumpliendo la restricción de no usar embeddings preentrenados. Limitaciones observadas del anotador en este corpus: spaCy asigna `Mood=Imp` a solo 63 tokens (detector de imperativo poco sensible) y no emite la etiqueta de dependencia `vocative`; por eso se combinaron tres detectores heurísticos y se reportó cada componente por separado. El NER quedó desactivado, de modo que la detección de nombres de persona usa una heurística de POS, mayúsculas y contexto de segunda persona.

Usos efectivos: (i) rasgos de segunda persona, imperativo, tercera persona plural y nombre propio como variables de contraste en el EDA y en el análisis de error; (ii) segmentación de rasgos por pliegue en los bloques densos H11/H12/H14 (sección 19). El bloque POS nunca se evaluó como representación independiente (el experimento E05 sobre sentimiento/emoción y el bloque estilométrico E06 no se ejecutaron), por lo que la literatura que sugiere su aporte en español (Garibo i Orts, 2019; García-Díaz et al., 2021, DOI 10.1016/j.future.2020.08.032) queda como hipótesis no probada en este proyecto.

## 11. Estilometría

Los rasgos estilométricos (longitud, puntuación, mayúsculas sostenidas, menciones, URLs, elongaciones, risas, emoticones) mostraron asociaciones **estadísticamente sólidas pero de efecto pequeño** con las tres etiquetas (V de Cramér entre 0,097 y 0,141 en los rasgos de estilo; la longitud por sí sola con delta de Cliff entre −0,141 y −0,369). La evidencia de la literatura sobre estilometría en textos cortos (Stamatatos, 2009, DOI 10.1002/asi.21001; Stamatatos, 2017, DOI 10.18653/v1/E17-1107) motivó la hipótesis H6, apoyada de forma parcial en el EDA.

En el modelo, los rasgos estilométricos entraron de forma implícita: los signos de puntuación repetidos y las mayúsculas se capturan parcialmente con `char_wb`, y las URLs/menciones quedan en el vocabulario de palabra. La experimentación explícita quedó cerrada en la práctica por dos resultados negativos: la variante que preserva mayúsculas (PRE-01) rinde −0,0229 y la eliminación de puntuación (PRE-07) es neutra (0,0000), lo que indica que la información de estilo útil ya está representada. El bloque estilométrico denso independiente (E06) no se ejecutó y se declara como trabajo pendiente.

## 12. Lexicones

**No se usó ningún lexicón externo.** La verificación de la auditoría confirmó que `data/external/lexicons/` está vacío, que no hubo descargas de recursos externos y que `reports/lexicon_catalog.csv` no existe, lo cual es consistente con la decisión D-007 (el catálogo solo era exigible «si se usan»). HurtLex (Bassignana et al., 2018, DOI 10.4000/books.aaccademia.3085) y los recursos afectivos tipo ML-SentiCon fueron contemplados como candidatos opcionales y finalmente **no se utilizaron ni se evaluaron**: los experimentos E02b (lexicón externo) y E04 (sentimiento/emoción) nunca se ejecutaron. La consecuencia honesta es que la hipótesis H5 (sentimiento/emoción aporta información complementaria) queda **sin probar**, pese a la evidencia favorable en HatEval español de Nina-Alcocer (2019) (`TFIDF+EMOTIONS`: EMR 0,711 vs 0,629 del baseline). Todo el léxico empleado en el proyecto es, por tanto, autocontenido: inducido de train, con contrato de inducción in-fold y trazabilidad en `reports/features_densas_resultados.json`.

## 13. Ingeniería de features

La representación final es deliberadamente simple y fue el resultado de una ablación sistemática:

```
texto crudo en minúsculas
  → FeatureUnion
      ├─ TfidfVectorizer(analyzer='word',    ngram_range=(1,2), min_df=2,
      │                  max_df=1.0, sublinear_tf=True)   → 4.024 features
      └─ TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2,
                         max_df=1.0, sublinear_tf=True)   → 23.407 features
  → 27.431 features
  → LogisticRegression(class_weight='balanced', C=2.0, solver='liblinear')  ×3
      (una por etiqueta HS, TR, AG)
  → umbrales por etiqueta: HS 0.44 / TR 0.51 / AG 0.46
```

Decisiones de ingeniería y su evidencia:

- **Sin eliminación de stopwords**: la variante PRE-02 rinde −0,0025 y en la literatura las palabras funcionales son señal (MineriaUNAM usa n-gramas de stopwords 2-4).
- **Sin lematización**: PRE-09 rinde −0,0162; el corpus sin diacríticos degrada el lematizador y los tweets pierden señal al normalizarse (H7 apoyada).
- **Sin reconstrucción de acentos**: prohibido introducir información externa (D-010); el corpus es ASCII por construcción.
- **`min_df=2`**: descarta n-gramas únicos manteniendo cobertura; las sondas con `min_df=5` (10.515 features) no mejoraron la transferencia (sección 19).
- **`sublinear_tf`**: amortigua el efecto de las repeticiones; presente en todas las variantes adoptadas.
- **Clasificadores independientes por etiqueta** con `class_weight='balanced'`: el desbalanceo moderado (máximo 2,34:1) no exigió remuestreo (H9/EDA); se descartó SMOTE/ADASYN.

Los bloques densos construidos para H11 (registro bromista, 22 features), H12 (objetivo individual vs colectivo, 15 features) y H14 (deshumanización inducida, 9 features) se estandarizaron con `StandardScaler` por pliegue y se concatenaron a la unión en 46 columnas; su resultado fue negativo y no entraron al modelo final (sección 19). El test nunca participó en ningún ajuste: la auditoría verificó que el vocabulario del artefacto final coincide exactamente con un ajuste desde cero solo con train (27.431 = 4.024 + 23.407).

## 14. Baselines

Se establecieron tres referencias y una sonda:

1. **Sonda E00 (identificación de la métrica)**: se envió la plantilla oficial todo-1 (`submissions/toxic_submission_v00_plantilla.csv`). LB observado **0,51494**. Los valores esperados con la distribución de train eran 0,541 (media column-wise de la F1 binaria), 0,543 (micro), 0,270 (macro sobre 6 pares) y 0,284 (samples/exact match). La observación descarta macro-6 y samples; la confirmación final llegó con el baseline: local 0,75778 (media column-wise) vs LB 0,75677 (diferencia −0,0010), mientras el macro-6 local era 0,8097. **Métrica oficial = F1 macro sobre las 3 columnas.**
2. **B1 (baseline clásico)**: TF-IDF word (1,2) min_df=2 sublinear + LR balanceada C=1.0, 4.024 features. OOF 0,7587 (sin post-proceso) / 0,7578 (con post-proceso jerárquico); por etiqueta HS 0,743 / TR 0,803 / AG 0,731; exact match 0,701; 1,14 s de entrenamiento. LB 0,75677.
3. **B5-DENSE (baseline denso)**: LR C=1.0 sobre los 46 rasgos densos H11+H12+H14 estandarizados, sin TF-IDF. OOF 0,6607. Sirve de control: los rasgos densos *per se* contienen señal, pero muy inferior a la representación de n-gramas.
4. **Sonda de la plantilla como suelo**: 0,51494 (todo positivo) delimita el comportamiento de un clasificador trivial en la métrica oficial.

El baseline B1 ya reveló el fenómeno que dominaría el proyecto: el modelo **subpredice positivos** con umbral 0,5 (HS predicho 0,373 vs prevalencia 0,437; AG 0,310 vs 0,382), lo que abrió la línea de umbrales por etiqueta.

## 15. Modelos clásicos

El barrido de la Parte C comparó clasificadores lineales sobre la representación ganadora (mismo split, mismo protocolo):

| ID | Modelo | F1 OOF | Δ vs B1 | Pliegues | Decisión |
|---|---|---:|---:|---:|---|
| MOD-02_BEST | ComplementNB(alpha=1.0) | 0,6888 | −0,0699 | 0/5 | Descartado |
| MOD-03a_BEST | LR C=0.5, liblinear, balanced | 0,7723 | +0,0135 | 5/5 | Adoptado como referencia |
| MOD-01_BEST | LinearSVC C=1.0, balanced | 0,7849 | +0,0262 | 5/5 | Adoptado como base de ensemble |
| REP-05 | LR C=1.0 sobre la unión | 0,7865 | +0,0278 | 5/5 | Adoptado |
| **MOD-03b_BEST** | **LR C=2.0, liblinear, balanced** | **0,7927** | **+0,0339** | **5/5** | **Modelo final** |
| MOD-04 | MOD-03b + post-proceso jerárquico | 0,7870 | +0,0283 | 5/5 | Descartado (−0,0056 vs su origen) |

El mensaje es consistente con Joachims (1998, DOI 10.1007/BFb0026683): en clasificación de texto con representaciones dispersas de alta dimensión, un modelo lineal regularizado es competitivo; la familia Naive Bayes queda muy por detrás al no modelar las interacciones de evidencia negativa. La regularización importa: C=2.0 supera a C=1.0 y a C=0.5 en OOF, pero el análisis posterior (D-013) mostraría que **más capacidad local no implicó mejor transferencia al test** — la mejora de C=2.0 sobre C=1.0 no se reprodujo en el leaderboard (v03 0,78306 vs v06 0,77227, aunque con umbrales distintos entre ambas).

## 16. Deep learning desde cero

La Fase 11 fue un **requisito metodológico**, no una palanca de rendimiento (D-014). Se entrenaron tres arquitecturas desde cero, con vocabulario y preprocesamiento ajustados únicamente con el pliegue de entrenamiento, cabezal multi-tarea de 3 logits, `BCEWithLogitsLoss` con `pos_weight` por etiqueta, AdamW (lr 1e-3, wd 1e-4), batch 32, máximo 30 épocas y early stopping estricto sobre la F1 de validación (paciencia 5, mejores pesos restaurados):

| ID | Arquitectura | Parámetros | F1 OOF | F1 por etiqueta (HS/TR/AG) | Gap train-val | Decisión |
|---|---|---:|---:|---:|---:|---|
| DL-01 | MLP sobre SVD(300) del TF-IDF unión, capas 256-128 | 110.339 | **0,7949 ± 0,0262** | 0,7838 / 0,8217 / 0,7790 | +0,1854 | Supera a MOD-03b en OOF (+0,0022); no adoptado (D-012/D-014) |
| DL-01-SEEDS | Idem, semillas 42/43/44 | 110.339 | 0,7947 ± 0,0011 | — | — | Estable entre semillas |
| DL-02 | CharCNN (embeddings de carácter dim 48, kernels 3 y 5) | 87.059 | 0,7410 ± 0,0097 | 0,7093 / 0,7763 / 0,7375 | +0,1807 | Descartado (peor que B1) |
| DL-03 | BiLSTM de palabra (embeddings dim 64, vocab 1.685, seq 40) | 233.411 | 0,7364 ± 0,0199 | 0,7237 / 0,7862 / 0,6992 | +0,1896 | Descartado (peor que B1) |

Lectura honesta: la mejor red (DL-01) alcanza 0,7949 de F1 OOF frente a 0,7927 del modelo clásico, una diferencia de +0,0022 que **la regla del proyecto no permite adoptar** (umbral +0,005) y que además carece de sonda de leaderboard: por D-012/D-014 no se gastó un envío. El MLP opera sobre una proyección SVD del mismo TF-IDF, de modo que su ventaja no proviene de una representación nueva sino de una frontera de decisión no lineal sobre la misma información. Los modelos secuenciales puros (CharCNN, BiLSTM) quedan por debajo incluso del baseline B1: con 1.731 ejemplos de entrenamiento, vocabularios diminutos y prohibición de embeddings preentrenados, el costo inductivo de las redes no se paga. Este resultado coincide con la advertencia de la literatura sobre el tamaño de datos en detección de odio (Wiegand et al., 2019) y con la evidencia de que los modelos lineales sobre n-gramas siguen siendo muy competitivos en corpus pequeños.

Nota técnica: AMP (autocast fp16) **no se usó** porque, medido en la GPU disponible (AMD RX 9070 XT, ROCm), el costo de `autocast` superaba la ganancia (BiLSTM: 3,42 s vs 7,48 s en 3 épocas). El protocolo de acceso a GPU compartida con el escritorio (D-008) se aplicó antes de cada job y quedó en `experiments/gpu_preflight.log`; el test nunca se cargó en el script de DL (0 ocurrencias de `read_test`, verificado por la auditoría).

## 17. Validación

El protocolo de validación evolucionó con la evidencia:

**Protocolo primario.** `StratifiedGroupKFold(k=5, shuffle=True, random_state=42)` con grupos = componentes conexas de similitud ≥ 0,85 (near-duplicados) y estratificación por la combinación HS×TR×AG. La reconstrucción independiente de la auditoría confirmó **0 grupos compartidos entre train y validación** en los 5 pliegues y **0 textos de validación con copia exacta en train** (normalizado). El vectorizador, el escalador y los léxicos inducidos se ajustan dentro de cada pliegue (`fit_transform` solo sobre el índice de entrenamiento); el test solo se usa con `transform`/`predict`.

**Control por pseudo-autor (D-015).** La auditoría final detectó que el control previsto en D-004 nunca se había ejecutado: con el split solo por near-duplicados, **17/17 pseudo-autores con ≥3 tweets cruzaban pliegues** (por ejemplo, `@youtube` aparecía en los 5). Al fusionar los grupos (near-duplicado + pseudo-autor) y reevaluar el modelo final con el mismo protocolo, la F1 OOF baja de 0,7927 a **0,7838** (−0,0088) y el **gap local-LB pasa de −0,0128 a −0,0039** comparado con el LB real de esa configuración (v02 = 0,77993). Desviación entre pliegues del control fusionado: 0,0187, frente a 0,0088 del split por near-duplicados. Conclusión: el split solo por near-duplicados era optimista porque un mismo usuario produce tweets con etiquetas correlacionadas; el protocolo recomendado para trabajo futuro es el agrupamiento fusionado. La selección del modelo final no se reabrió (D-014), pero el hallazgo se documenta como la explicación principal del gap.

**Evaluación de umbrales y ensembles.** Los umbrales se optimizaron solo sobre probabilidades OOF (rejilla 0,20–0,80 paso 0,01, por etiqueta) y la ganancia optimista (+0,0063) se contrastó con una verificación anidada independiente: **+0,0014 de media, 2/5 pliegues positivos**. Los ensembles se evaluaron con protocolo leave-one-fold-out (pesos/meta-modelo ajustados en 4 pliegues y evaluados en el restante), con la limitación declarada de que las probabilidades base son OOF compartidas.

**Auditoría de fuga.** Aprobada con observaciones: sin uso de corpus/etiquetas externas, sin embeddings preentrenados, sin selección de features/umbrales sobre test, sin procesos de envío automático. Dos hallazgos relevantes: (i) la brecha del pseudo-autor (arriba); (ii) los grupos de near-duplicados guardados son conservadores. El vocabulario del artefacto final coincide exactamente con un ajuste solo-train; las submissions v01, v02 y v03 se reprodujeron **bit a bit** (sha256 idénticos) en una copia aislada.

**Reproducibilidad.** Semillas `random_state=42` en todos los scripts clásicos; DL con semillas 42/43/44; resultados deterministas (dos ejecuciones de `04_refit_submissions_v02_v03.py` en copia aislada produjeron la misma métrica y las mismas submissions). 6/6 modelos joblib, 5/5 JSON de métricas y 3/3 NPZ OOF cargan con formas correctas.

## 18. Experimentos

Se registraron **51 experimentos** en `experiments.csv` (52 líneas con cabecera) con pregunta, hipótesis, preprocesamiento, representación, modelo, hiperparámetros, protocolo de validación, métrica, interpretación y decisión. La siguiente tabla sintetiza los hitos por fase:

| Fase | ID | Experimento | F1 OOF | Δ vs referencia | Decisión |
|---|---|---|---|---:|---|
| Métrica | E00 | Sonda de plantilla todo-1 | — | LB 0,51494 | Métrica identificada (D-009) |
| Baseline | B1 | TF-IDF word + LR | 0,7587 | — | Referencia adoptada |
| Preproc. | PRE-03 | URLs → token `url` | 0,7623 | +0,0035 | Descartada (< 0,005) |
| Preproc. | PRE-09 | Lematización spaCy | 0,7425 | −0,0162 | Descartada |
| Repres. | REP-05 | Unión word+char | 0,7865 | +0,0278 | Adoptada |
| Modelos | MOD-03b | LR C=2.0 sobre la unión | **0,7927** | +0,0339 | **Modelo final** |
| Post-proc. | MOD-04 | HS=0 ⇒ TR=0, AG=0 | 0,7870 | −0,0056 | Descartada (H8) |
| Umbrales | TH-01 | Umbrales OOF 0,44/0,51/0,46 | 0,7989 | +0,0063 (optimista) | Adoptada como apuesta (D-012) |
| Ensembles | ENS-01..04 | Votación, pesos, stacking, mejor par | 0,7906–0,7985 | máx. +0,0058 (máx. de 12) | Descartados (< +0,005 anidado; v04 LB 0,77510) |
| Densas | HIB-H12 | Unión + objetivo individual/colectivo | 0,7836 | −0,0091 | Descartada (H12) |
| Densas | HIB-H11 | Unión + registro bromista | 0,7713 | −0,0213 | Descartada (H11) |
| Densas | HIB-H14 | Unión + deshumanización | 0,7630 | −0,0297 | Descartada (H14) |
| Densas | HIB-UNION | Unión + 46 densas | 0,7610 | −0,0316 | Descartada |
| Densas | B5-DENSE | Solo 46 densas | 0,6607 | −0,1319 | Descartada |
| Optuna | OPT-01 | Mejor trial (50 trials, 28 completos) | 0,8012 (0,8039 con umbrales) | +0,0086 | Adoptado localmente; v05 LB 0,77602 → descartado |
| Robustez | R1-C1 | C=1.0 | 0,7929 | +0,0002 | v06 LB 0,77227 → descartada |
| Robustez | R2-MINDF5 | min_df=5, C=1.0 | 0,7993 | +0,0064 | v07 LB 0,77034 → descartada |
| DL | DL-01 | MLP sobre SVD(300) | 0,7949 | +0,0022 | Requisito metodológico; no adoptado |
| DL | DL-02/03 | CharCNN / BiLSTM | 0,7410 / 0,7364 | −0,052 / −0,056 | Descartadas |
| Validación | PA-01 | Control por pseudo-autor | 0,7838 | −0,0088 | Protocolo recomendado (D-015) |
| v2 | S08 | Sonda v08: meta de stacking v2 | 0,7874 (dev) | LB 0,78163 | No adoptada (−0,0014 vs v03) |
| v2 | CIERRE | Investigación v2 (24 líneas, P0–P7) | — | — | Sin mejora transferible; modelo final V03 |

Patrón general: de las más de 50 configuraciones evaluadas, **una sola familia** (unión word+char con LR lineal y umbrales) se adoptó como modelo final; todas las vías alternativas —preprocesamiento lingüístico, features densas, ensembles, optimización de hiperparámetros, control de capacidad y redes desde cero— produjeron mejoras locales nulas, negativas o no transferibles. Este patrón no es un fracaso del proceso sino su resultado principal: la representación de n-gramas con un clasificador lineal regularizado es muy difícil de superar en este régimen de datos y restricciones.

## 19. Ablación

La ablación permitió separar qué componente aporta y qué no. Componentes evaluados con el mismo protocolo agrupado:

| Componente | Variante | Efecto medido | Veredicto |
|---|---|---|---|
| Preprocesamiento (10 variantes) | Mejor: URLs → `url` (PRE-03) | +0,0035 (< 0,005) | Descartado |
| Preprocesamiento | Lematización | −0,0162 | Descartado (H7 apoyada) |
| Preprocesamiento | Preservar mayúsculas | −0,0229 | Descartado |
| Preprocesamiento | Quitar stopwords | −0,0025 | Descartado |
| Preprocesamiento | Eliminar puntuación | 0,0000 | Neutro |
| Representación | char_wb sola | −0,0021 vs word sola | H1 no confirmada en forma fuerte |
| Representación | Unión word+char | +0,0278 (5/5 pliegues) | **Adoptado** (H4) |
| Capacidad | C=2.0 → C=1.0 | OOF −0,0062; LB −0,0108 | Descartado (D-013) |
| Capacidad | min_df 2 → 5 | OOF +0,0004 con umbrales; LB −0,0127 | Descartado (D-013) |
| Post-proceso | HS=0 ⇒ TR=0, AG=0 | −0,0056 / −0,0004 (solo TR) | Descartado (H8) |
| Umbrales | Por etiqueta vs 0,5 | OOF +0,0063; LB +0,0031 pareado; anidado +0,0014 | Adoptado (D-012, H9) |
| Densas | H11 / H12 / H14 / unión | −0,0213 / −0,0091 / −0,0297 / −0,0316 | Descartado (H11, H12, H14) |
| Ensemble | 4 estrategias | máx. +0,0023 anidado; v04 LB −0,0080 vs v03 | Descartado (H10) |
| DL | 3 arquitecturas | máx. +0,0022 OOF sin sonda | No adoptado |
| Validación | Grupos fusionados con pseudo-autor | −0,0088 (calibración, no rendimiento) | Adoptado como protocolo |

Un control especialmente informativo: al poner las 46 columnas densas a cero dentro del pipeline híbrido, la métrica reproduce exactamente 0,792659 (la de MOD-03b). Es decir, **la pérdida de los híbridos es atribuible a las features densas, no a la concatenación**, lo que descarta errores de implementación como causa del resultado negativo. De forma análoga, el control REF-MOD03b de las sondas de robustez reprodujo 0,792659 exacto y probabilidades idénticas a `features/oof_mod03b.npz` (diferencia máxima 0.0e+00): el arnés experimental es fiel.

## 20. Análisis de error

El análisis de error (`reports/error_analysis.md`) usó las probabilidades OOF del modelo final con los umbrales adoptados (HS 0,44 / TR 0,51 / AG 0,46). Errores totales: **425 falsos positivos y 375 falsos negativos** sobre 3×1.731 decisiones; 251 ejemplos fallan una etiqueta, 123 fallan dos y **101 (5,8 %) fallan las tres**. Con umbral 0,5 el balance era 349 FP / 445 FN: los umbrales trasladan error de FN a FP.

Matrices de confusión (escenario de umbrales adoptados):

| Etiqueta | Umbral | TP | FP | FN | TN | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HS | 0,44 | 614 | 185 | 142 | 790 | 0,769 | 0,812 | **0,790** |
| TR | 0,51 | 431 | 89 | 88 | 1123 | 0,829 | 0,830 | **0,830** |
| AG | 0,46 | 517 | 151 | 145 | 918 | 0,774 | 0,781 | **0,777** |

TR es la etiqueta mejor resuelta; HS la de más errores absolutos (327); AG la de menor F1. La regla estructural `HS=0 ⇒ TR=0` vale −0,0004 y su versión completa −0,0049: no es una palanca (H8 cerrada).

**Dónde falla el modelo.** Las 5 rebanadas más débiles (n ≥ 40, excluyendo artefactos in-sample del léxico):

| # | Rebanada | n | F1 HS | F1 TR | F1 AG | F1 medio | Δ vs global |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `tiene_hashtag=1` | 161 | 0,622 | 0,545 | 0,590 | **0,586** | −0,213 |
| 2 | `tiene_pregunta=1` | 158 | 0,682 | 0,653 | 0,563 | 0,633 | −0,166 |
| 3 | `tiene_referencia_colectiva=1` | 418 | 0,697 | 0,644 | 0,576 | 0,639 | −0,160 |
| 4 | cuartil de longitud Q4 | 432 | 0,696 | 0,629 | 0,611 | 0,645 | −0,154 |
| 5 | `tiene_mencion=0` | 909 | 0,703 | 0,707 | 0,681 | 0,697 | −0,102 |

Regiones del corpus: `responde_mencion_inicial` (n=701) F1 0,876; `regimen_noticia` (URL o hashtag, n=599) F1 0,702; `no_responde_con_colectiva` (n=318) F1 0,597. La agresividad que se escapa es contra grupos: el **45,5 % de los FN de AG tiene referencia colectiva frente al 13,9 % de los TP**. El error de TR vive fuera de las respuestas: F1_TR 0,875 con mención inicial vs 0,597 sin mención con colectiva.

**Patrones cualitativos** (revisión de 90 muestras en `reports/eda/error_samples_para_revision.csv`):

1. **Insulto interpersonal en registro de broma o afiliación** (FP dominante de HS, TR y AG): de 526 textos con insulto, mención inicial y sin referencia colectiva, **213 (40,49 %) están etiquetados (0,0,0)**. El modelo dispara con probabilidad 0,87–0,97 textos como `@CamilaArvayo Cllate todo te queda bien zorra te amo`, porque el vocativo, los posesivos afectivos («mi novia», «mi perra»), la risa (`JAJAJA`) y las marcas de juego desactivan la lectura tóxica en la anotación, pero un modelo de bolsas de n-gramas suma evidencia de insulto y no representa conjunciones con marcadores de afiliación.
2. **Uso metalingüístico: cita o reproducción del insulto** (FP y FN): el modelo lineal reacciona al léxico citado tanto cuando la cita es inocua como cuando la cita es el vehículo del odio.
3. **Régimen noticia o atribución a terceros** (FN dominante): textos largos con URL que describen violencia contra un grupo; varios describen a un individuo concreto, lo que refuerza la lectura de D-005.
4. **Odio implícito sin insulto del léxico**: estereotipo («El Arabe es un flojo»), miedo estigmatizante («m da miedo cuando me sigue un rabe»), deshumanización («una linda gallina que acabar en un KFC»). El modelo necesita el insulto explícito para activar HS/AG.
5. **Frontera de anotación no observable en el texto**: casos de superficie casi idéntica con etiquetas opuestas; el 40,49 % de la región de insulto interpersonal es (0,0,0). Este límite acota el techo alcanzable con cualquier modelo de texto.

Del análisis surgieron cinco hipótesis nuevas (H11–H15) con experimento y criterio; su ejecución posterior fue predominantemente negativa (H11 −0,0213; H12 −0,0091; H14 −0,0297), y **H13 (cita/atribución) y H15 (umbrales por rebanada) nunca se ejecutaron**; se declaran como trabajo futuro.

## 21. Modelo final

**MOD-03b_BEST + umbrales por etiqueta** (decisión D-014, cerrada el 2026-09-23):

- **Representación**: `FeatureUnion` de TF-IDF word (1,2) min_df=2 max_df=1.0 sublinear_tf (4.024 features) + TF-IDF char_wb (3,5) min_df=2 max_df=1.0 sublinear_tf (23.407 features) = **27.431 features**.
- **Modelo**: tres `LogisticRegression(class_weight='balanced', C=2.0, solver='liblinear', random_state=42, max_iter=1000)`, una por etiqueta, sin post-proceso jerárquico.
- **Umbrales de decisión** (optimizados solo sobre OOF): HS **0,44**, TR **0,51**, AG **0,46**.
- **Métricas OOF**: F1 macro columnas **0,7989** (0,7927 con umbral 0,5); F1 por etiqueta HS 0,7897 / TR 0,8296 / AG 0,7774; macro-6 0,8356; micro 0,7898; exact match 0,7314 (umbral 0,5); desviación entre pliegues 0,0088.
- **Leaderboard**: **0,78306** (submission v03), el mejor de 8 envíos; gap local-LB −0,0158 con el split original, **−0,0039** con el control por pseudo-autor (D-015).

**Por qué se mantienen los umbrales pese a la verificación anidada modesta**: el contraste pareado v02 (umbral 0,5) vs v03 (umbrales) en el mismo modelo dio **+0,0031 real en el leaderboard**, dentro del rango entre la estimación anidada (+0,0014) y la optimista (+0,0063). Se prohibió re-optimizar umbrales contra el LB. La submission v02 se conserva como control metodológico (LB 0,77993).

**Artefactos**: `models/mod03b_lr_union.joblib` (vectorizador + 3 modelos + umbrales), `features/oof_mod03b.npz` (probabilidades OOF), `submissions/toxic_submission_v03_lr_union_thr.csv` (sha256 `7d7478f271e6edfd…`), script `experiments/04_refit_submissions_v02_v03.py`. La auditoría verificó que el artefacto reproduce v03 exactamente (0 filas distintas) y que la submission se reconstruye byte a byte desde el script en copia aislada.

**Plateau (D-014)**: los 6 modelos de la familia union evaluados en el LB caen en la banda **0,77034–0,78306**, un rango de 0,0127 = **0,99 errores estándar** (SE binomial ≈ 0,0129 con n=1.154). Las diferencias entre variantes son indistinguibles del ruido de muestreo; el único contraste pareado fiable es el de umbrales. Por eso se detuvo el sondeo del leaderboard y se redirigió el esfuerzo a los entregables académicos y a completar la Fase 11 como requisito metodológico.

## 22. Submission

Se enviaron 8 submissions, todas validadas contra la plantilla oficial (columnas `id,HS,TR,AG`, 1.154 filas, ids idénticos y en orden, valores en {0,1}, sin nulos ni duplicados) y **enviadas manualmente por Carlos** (0 llamadas a la API/CLI de envío en el código). Registro completo en `submissions/leaderboard_log.csv`:

| Envío | Modelo | Local (OOF) | Kaggle | Gap | Positivos HS/TR/AG (%) |
|---|---|---:|---:|---:|---|
| v00 plantilla | Todo 1 | — | 0,51494 | — | 100 / 100 / 100 |
| v01 baseline | TF-IDF word + LR | 0,75778 | 0,75677 | −0,0010 | 37,4 / 27,9 / 31,0 |
| v02 union u=0,5 | MOD-03b, umbral 0,5 | 0,79270 | 0,77993 | −0,0128 | 38,0 / 29,3 / 34,2 |
| **v03 union + umbrales** | **MOD-03b + 0,44/0,51/0,46** | **0,79890** | **0,78306** | −0,0158 | 43,9 / 28,9 / 36,6 |
| v04 ensemble mediana | ENS-04 (M2+M3), umbrales medianos | 0,79850 | 0,77510 | −0,0234 | 46,5 / 30,2 / 40,5 |
| v05 optuna | OPT-01 + umbrales | 0,80390 | 0,77602 | −0,0279 | 41,0 / 29,7 / 35,8 |
| v06 robustez C=1.0 | MOD-03b C=1.0 + umbrales | 0,79290 | 0,77227 | −0,0206 | 44,5 / 28,2 / 38,6 |
| v07 robustez min_df=5 | min_df=5, C=1.0 + umbrales | 0,79930 | 0,77034 | −0,0290 | 45,1 / 29,2 / 35,4 |

Hallazgos de la fase de submission:

- La sonda E00 fijó la métrica y reveló que el test tiene del orden de **7 % menos positivos** que train (factor ≈ 0,93): el valor 0,51494 está por debajo del 0,541 esperado con la prevalencia de train.
- **El mejor OOF no fue el mejor LB**: OPT-01 (0,8039 local) obtuvo 0,77602; MOD-03b (0,7989) obtuvo 0,78306. El gap local-LB creció con la capacidad del modelo (B1 −0,0010; MOD-03b −0,0128/−0,0158; ENS-04 −0,0234; OPT-01 −0,0279).
- **El gap train-OOF no predice el gap local-LB** (D-013 refutada): reducir C e incrementar min_df redujo el gap train-OOF (0,1818 → 0,1635 → 0,1272) pero empeoró el gap local-LB (−0,0206 y −0,0290).
- El ensemble con umbrales medianos predijo más positivos que la prevalencia de train y cayó (0,77510); el patrón observado es que sobredisparar la prevalencia del test penaliza, aunque la evidencia es insuficiente para convertirlo en regla (v05, con prevalencia cercana a la estimada, tampoco superó a v03).

**Estado final**: la submission vigente y de respaldo es **v03** (LB 0,78306). Fallback secundario: v02 (control sin umbrales, 0,77993); último recurso v01 (0,75677). Las variantes v04–v07 permanecen en disco como evidencia y no deben reenviarse.

## 23. Limitaciones

1. **Tamaño y calidad del corpus**: 1.731 ejemplos de entrenamiento para 27.431 features y tres tareas; 101 ejemplos (5,8 %) fallan las tres etiquetas y la frontera de anotación entre insulto interpersonal y odio no es observable en el texto (40,49 % de la región de insulto + mención sin colectivo está etiquetada como (0,0,0)). Cualquier modelo de texto tiene un techo acotado por esta ambigüedad (Wiegand et al., 2019; Röttger et al., 2021, DOI 10.18653/v1/2021.acl-long.4).
2. **Corpus normalizado a ASCII**: sin tildes ni eñe, con 6 bytes `0xBF` en test; parte de la señal morfológica del español es irrecuperable y no se debe reconstruir con recursos externos (D-010).
3. **Gap local-LB**: la cota local fue optimista entre −0,001 (B1) y −0,029 (v07). Con el protocolo de grupos fusionados (D-015) el gap se reduce a −0,0039 para el modelo final, pero ese protocolo no se usó para la selección original.
4. **Plateau**: las diferencias entre variantes de la familia union (0,77034–0,78306) equivalen a ~1 error estándar; el ranking del leaderboard a este nivel es en gran parte ruido de muestreo.
5. **Pseudo-autores**: 17 pseudo-autores (77 tweets, 4,4 %) cruzaban pliegues en el split original; la validación sin ellos era optimista en ~0,0088. El protocolo recomendado (grupos fusionados) se documentó, pero no se volvió a seleccionar modelo con él.
6. **TR sin resolver (D-005)**: la semántica de TR sigue ambigua entre la definición oficial y el esquema HatEval; la revisión humana prevista no se completó. El modelo usa la etiqueta tal como aparece.
7. **Hipótesis y experimentos no ejecutados**: H5 (sentimiento/emoción) con E02b/E04/E05; H13 (cita/atribución); H15 (calibración por rebanada); el bloque estilométrico/POS E06. Se declaran como alcance no cubierto, no como resultados negativos.
8. **DL-01 sin sonda de leaderboard**: superó a MOD-03b en OOF (+0,0022) y no se envió; la omisión es una decisión explícita por D-012/D-014 (regla de +0,005 y cierre de la fase competitiva), no una omisión silenciosa.
9. **Near-duplicados train-test**: 10 textos de test tienen near-duplicados en train; la partición agrupada lo mitiga, pero el escenario del LB incluye esos casos y la estimación local no es perfectamente comparable.
10. **Estimación indirecta de la prevalencia del test**: el factor ≈0,93 se infiere de la sonda E00 y de los gaps, no de las etiquetas de test (prohibido usarlas).
11. **Una sola semilla de partición**: `random_state=42` en todos los experimentos clásicos; la robustez se evaluó vía variantes y el control por pseudo-autor, no vía múltiples particiones (los ensembles/densas sí usaron esquemas anidados).
12. **30 % de la nota depende del notebook**: el entregable `notebooks/10_final_pipeline.ipynb` (31 celdas, todas las de código ejecutadas) existe y consolida la narrativa, pero su evaluación es independiente de este informe.
13. **Límites evidenciados en la fase v2**: los deltas locales que respaldaron al candidato (+0,0081 en desarrollo; +0,0180 en el holdout de 248 tweets) son del orden del error estándar del test (~0,013 con n=1.154; el holdout tiene EE ~0,03), por lo que no garantizaban transferencia; el gap dev-LB de la familia union es negativo (−0,013 a −0,016) y se confirmó en todas sus variantes; y con 1.154 filas la comparación en el test no distingue V03 del meta (diferencia −0,0014 frente a un error estándar ~0,013), de modo que la única lectura que resiste es que ambos son indistinguibles en el test.

## 24. Conclusiones

1. **La métrica correcta se determinó empíricamente**, no por supuesto: F1 macro sobre las 3 columnas (media column-wise de la F1 binaria), confirmada por la sonda E00 (0,51494) y por el baseline (local 0,75778 vs LB 0,75677). El protocolo local reproduce el LB con error ~0,001 en configuraciones de baja capacidad.
2. **La representación de n-gramas con un clasificador lineal regularizado es el núcleo ganador** en este régimen de datos y restricciones: unión word(1,2)+char_wb(3,5) con LR C=2.0 aporta +0,0339 sobre el baseline B1 con mejora en 5/5 pliegues, y ninguna de las más de 50 configuraciones evaluadas logró superarla de forma transferible.
3. **Los umbrales por etiqueta aportaron una mejora real, pequeña y pareada** (+0,0031 en el LB, mismo modelo): es el único contraste del leaderboard que supera el ruido, coherente con la literatura de umbral óptimo para F1 (Lipton et al., 2014; Nan et al., 2012).
4. **Los resultados negativos son evidencia de primera clase**: preprocesamiento lingüístico, features densas, ensembles, Optuna, control de capacidad y DL desde cero no superaron la barrera pre-registrada; varios de ellos (los híbridos densos) fueron controlados experimentalmente para descartar artefactos de implementación.
5. **Dos hallazgos metodológicos transferibles**: (i) el gap train-OOF no predice el gap local-LB (D-013 refutada) porque el optimismo de la validación agrupada es de otra naturaleza que la memorización in-sample; (ii) el pseudo-autor explica la mayor parte del gap optimista del protocolo (D-015: gap −0,0128 → −0,0039), por lo que el agrupamiento debe fusionar near-duplicados y autores.
6. **La fase competitiva alcanzó su plateau** con los datos y restricciones disponibles: 6 modelos en una banda de ~1 error estándar. La decisión D-014 de detener el sondeo fue la respuesta correcta al riesgo de sobreajustar el leaderboard.
7. **Las restricciones se cumplieron íntegramente** y quedaron verificadas por auditoría independiente: sin transformers/LLM, sin embeddings preentrenados, sin datos externos, test nunca usado para entrenar, submissions en formato válido, resultados reproducibles bit a bit. Veredicto GO.
8. **El resultado final** (LB 0,78306, OOF 0,7989 con umbrales y 0,7838 con el protocolo corregido por pseudo-autor) es consistente con el estado del arte clásico en español: el mejor sistema sin transformers de HatEval español logró macro-F1 0,73 (Argota Vega et al., 2019) en un corpus binario de odio, y este proyecto opera en tres etiquetas simultáneas con métrica macro.

## 25. Trabajo futuro

1. **Adoptar el protocolo de grupos fusionados (near-duplicado + pseudo-autor)** como estándar de validación en cualquier extensión del trabajo; reportar siempre ambas estimaciones (D-015).
2. **Ejecutar las hipótesis pendientes** con la regla de decisión pre-registrada: H13 (features de cita/atribución para el régimen noticia), H15 (umbrales por rebanada, como experimento de cierre), y H5/E02b/E04/E05 (sentimiento/emoción y lexicones externos con catálogo y licencia, manteniendo una variante autocontenida).
3. **Atacar los dos regímenes de error por separado**: (i) registro de respuesta con interacciones explícitas insulto × afiliación (la línea H11 necesita otra formulación: regularización por bloque, modelos con conjunciones o calibración condicionada por subgrupo); (ii) régimen noticia/atribución con features de metalenguaje y estructura del titular.
4. **Explorar el MLP (DL-01) como fuente de diversidad real** frente al lineal: pese a que su OOF no es adoptable sin sonda, su frontera no lineal podría ensamblarse mejor con la del LR que los ensembles de modelos correlacionados, previa validación anidada rigurosa.
5. **Calibración probabilística** (Platt/Isotonic) y estudio de costos asimétricos si el caso de uso final pondera de forma distinta los FP y los FN de odio.
6. **Validación externa controlada**: repetir el pipeline en otro corpus español de odio con el mismo esquema, verificando que la representación y los umbrales transfieren, sin usar etiquetas externas en el entrenamiento del modelo de la competencia.
7. **Anotación y estudio de la frontera**: cuantificar el techo real con una muestra doblemente anotada en la región de insulto interpersonal (donde el desacuerdo entre anotadores acota el rendimiento alcanzable) y cerrar la revisión de D-005 con anotadores humanos.
8. **Notebook y documentación**: mantener el notebook como fuente reproducibilidad de punta a punta (regeneración de la submission v03 verificada bit a bit) y ampliar la discusión didáctica de los resultados negativos, que son el mayor activo formativo del proyecto.
9. **Ejecutar las líneas del backlog v2 no cubiertas** (5 de 24; las demás se evaluaron y cerraron en la fase v2): calibración de probabilidades (Platt/isotonic en OOF anidado, E2-14), encoder de entidades (target encoding OOF de menciones/hashtags/dominios, exigiendo cobertura > 20 %, E2-13), especialistas por rebanada (E2-20), umbrales por rebanada con shrinkage (E2-16, H15) y espacios de features por etiqueta (E2-21), cada una con la regla de adopción pre-registrada de D-016.
10. **Adoptar el protocolo de validación corregido de la v2 en cualquier trabajo derivado**: grupos fusionados (near-duplicado UNION pseudo-autor) + holdout congelado con presupuesto explícito de evaluaciones + bootstrap por tweet, reportando siempre la incertidumbre de los deltas y evitando tratar el máximo observado como objetivo reproducible (winner's curse).

## 26. Investigación posterior (fase v2)

Fecha de la fase: 2026-09-25. Estado: **concluida sin mejora robusta transferible al test** (condición de terminación B del plan v2: las fuentes razonables de señal fueron evaluadas y documentadas). Fuentes: `reports/advanced_research_v2.md` (informe de cierre), `DECISIONS.md` (D-016 a D-020), `tasks/todo.md` (gates P0–P7) y `notebooks/10_final_pipeline.ipynb` (secciones 9–11 y 14).

Contexto: tras cerrar la fase competitiva (D-014), se ejecutó una segunda ronda de investigación de **24 líneas** para intentar superar el LB 0,78306 de V03. A diferencia de la ronda histórica, toda comparación se hizo con un protocolo de validación corregido y una regla de adopción pre-registrada.

### 26.1 Protocolo de validación corregido (D-016)

- **Grupos** = near-duplicado **UNION** pseudo-autor (grupos fusionados), centralizados en `src/splits.py`.
- **Holdout congelado**: 1 pliegue de `StratifiedGroupKFold(n_splits=7, seed=42)` = **248 tweets** (los 6 combinaciones de etiquetas representadas), con sha256 verificado (`e5b62e5f...`) y presupuesto máximo de **5 evaluaciones**, de las cuales se usaron **2**. El resto del train es el **desarrollo (1.483 tweets)**.
- **Métrica primaria**: `corrected_cv` sobre el desarrollo (`historical_cv` se reporta para continuidad). Referencia v2: **V03 con umbrales = 0,7772** (M2 con umbral 0,5 = 0,7718; por etiqueta HS 0,762 / TR 0,816 / AG 0,754).
- **Regla de adopción pre-registrada**: corrected_cv ≥ +0,005 con mejora en ≥ 3/5 pliegues, o complementariedad demostrada (`ensemble_gain` ≥ 0,0015 con IC bootstrap > 0). **Bootstrap por tweet** (2.000 remuestreos) obligatorio para todo candidato.

### 26.2 Resultados por paquete (P0–P7)

| Paquete | Línea | Mejor variante | corrected_cv | Δ vs V03 (0,7772) | Decisión |
|---|---|---|---:|---:|---|
| P0 | Scaffold: holdout, OOF corregido, gap analysis | V03 reevaluada | 0,7772 | — | Referencia v2 |
| P1 | Cadenas de etiquetas (E2-01) | HS_AG_TR | 0,7744 | −0,0028 (1/5) | Cerrada (IC incluye 0) |
| P1 | Meta de dependencia de etiquetas (E2-02) | LR C=0,1 sobre 44 features | **0,7853** | **+0,0081** (4/5) | Adoptada como candidato (IC [+0,0021, +0,0248]) |
| P2 | Targeting estructural (E2-05/06) | union + n-gramas simbólicos | 0,7487 | −0,0285 (1/5) | Cerrada; recall de AG en la rebanada colectiva 0,452 → 0,478 |
| P3 | POS / dependencias / morfología sparse (E2-03/04) | union + POS | 0,7656 | −0,0116 (0/5) | Cerrada (dependencias 0,7653; morfología 0,7567) |
| P4 | Cita/atribución (E2-07) y negación (E2-08) | union + negación | 0,7657 | −0,0115 (0/5) | Cerrada (cita/atribución 0,7554) |
| P5 | Emoción (E2-09) | union + afecto | 0,6330 | −0,1442 (0/5) | Cerrada |
| P5 | NB-SVM (E2-15) | NB-union + LR | 0,7612 | −0,0160 (2/5) | Cerrada (gain +0,0038, IC [−0,0081, +0,0162]) |
| P5 | Multicanal ortográfico (E2-10) | raw + colapsado | 0,7534 | −0,0238 (0/5) | Cerrada |
| P6 | Stacking v2 (E2-17/18) | META-BASES-RIDGE alpha=3 | **0,7874** | **+0,0102** (3/5) | Adoptado por complementariedad (gain +0,0089, IC [+0,0019, +0,0160]); los canales no aportan (Δ −0,0006); recall de AG colectiva 0,452 → 0,574 |
| P7 | Holdout congelado | meta Ridge alpha=3 | 0,8465 (V03 0,8285) | +0,0180 pareado | GO (IC [+0,0003, +0,0372]) |
| P7 | Sonda de leaderboard | v08 meta v2 | — | LB 0,78163 | **No adoptada** (−0,0014 vs v03) |
| v09 C-EMO | V03 + emoción (Ridge α=0,3) | 0,78094 | 0,77895 | −0,0020 |
| v10 C-SYN | V03 + sintaxis (Ridge α=1) | 0,77931 | 0,77978 | +0,0005 |
| v11 C-EMO-SYN | V03 + emoción + sintaxis (Ridge α=1) | 0,77966 | 0,78150 | +0,0018 |

Lectura: las líneas de representación (targeting, sintaxis, cita/negación, emoción, NB-SVM, ortografía) se cerraron sin mejora, repitiendo el patrón histórico de que la concatenación de canales degrada; el único valor apareció en la **fusión tardía a nivel meta**, que mejoró en desarrollo y en el holdout congelado pero no en el test.

### 26.3 El candidato y su validación

El candidato v2 es el **meta-modelo de dependencia de etiquetas** (RidgeClassifier alpha=3, balanced, sobre 44 meta-features: probabilidades, márgenes y entropías OOF de los 4 bases de texto más 8 rasgos estructurales), adoptado en P6 por complementariedad.

- **Desarrollo**: corrected_cv **0,7874** (+0,0102 vs V03; 3/5 pliegues). Su ventaja de nivel frente al meta de P1 (0,7853) es ruido (IC [−0,0054, +0,0093] incluye 0); lo que lo respalda es la complementariedad (`ensemble_gain` +0,0089, IC [+0,0019, +0,0160]).
- **Holdout congelado** (248 tweets, 2/5 evaluaciones usadas): V03 **0,8285** (HS 0,857 / TR 0,849 / AG 0,779) frente al meta **0,8465** (HS 0,862 / TR 0,857 / AG 0,820); delta pareado **+0,0180** con IC95 [+0,0003, +0,0372]. El holdout resultó un pliegue más fácil que cualquier pliegue del desarrollo (0,83/0,85 vs 0,774/0,787).
- **Auditoría (`ml-reviewer`)**: reprodujo los números exactos, verificó 0 grupos/textos compartidos con el desarrollo, confirmó que no hay fuga en el stacking y que la submission v08 coincide con el pipeline declarado.
- **Sonda de leaderboard v08 = 0,78163** (positivos predichos HS 44,8 % / TR 32,5 % / AG 38,6 %), por debajo de v03 (0,78306) en −0,0014. El meta mejoró en desarrollo (+0,0102) y en el holdout congelado (+0,0180) pero **no transfirió al test**; su gap dev-LB fue −0,0058.

### 26.4 Lectura honesta del leaderboard

La fase v2 y la fase final de candidatos dejaron **12 sondas** en `submissions/leaderboard_log.csv`:

| Sonda | Modelo | Local (OOF) | Kaggle | Gap |
|---|---|---:|---:|---:|
| v00 plantilla | Todo 1 | — | 0,51494 | — |
| v01 baseline | TF-IDF word + LR | 0,75778 | 0,75677 | −0,0010 |
| v02 union u=0,5 | MOD-03b, umbral 0,5 | 0,79270 | 0,77993 | −0,0128 |
| v03 union + umbrales | MOD-03b + 0,44/0,51/0,46 | 0,79890 | 0,78306 | −0,0158 |
| v04 ensemble mediana | ENS-04 | 0,79850 | 0,77510 | −0,0234 |
| v05 optuna | OPT-01 + umbrales | 0,80390 | 0,77602 | −0,0279 |
| v06 robustez C=1.0 | MOD-03b C=1.0 + umbrales | 0,79290 | 0,77227 | −0,0206 |
| v07 robustez min_df=5 | min_df=5, C=1.0 + umbrales | 0,79930 | 0,77034 | −0,0290 |
| v08 meta v2 | Meta Ridge alpha=3 (stacking v2) | 0,78740 | 0,78163 | −0,0058 |

- **La media de las 7 sondas de la familia union (v02–v08) es 0,7769** y el rango completo de esas sondas (0,77034–0,78306) es de 0,0127, es decir **≈ 0,99 errores estándar** (SE ≈ 0,0129 con n = 1.154): el ranking del leaderboard a este nivel es, en gran parte, ruido de muestreo.
- El máximo observado (0,78306 de v03) está **sesgado hacia arriba por selección (winner's curse)**: se eligió como máximo de 12 mediciones ruidosas, y tratarlo como objetivo reproducible sería un error.
- Los únicos contrastes pareados fiables son los que comparten casi todo el modelo: **v02 vs v03 (+0,0031** por umbrales) y **v03 vs v08 (−0,0014)**; ambos muy por debajo del error estándar.
- La sonda v08 confirma que la ventaja local del meta (respaldada incluso por un holdout congelado) no predijo la transferencia; la única lectura que resiste es que **V03 y el meta son indistinguibles en el test**.
- Las tres sondas de la fase final (v09–v11, metas restringidos sobre V03 + canales) obtuvieron 0,77895 / 0,77978 / 0,78150: las tres por debajo de V03 y dentro de la banda; el ranking local volvió a no predecir el del LB (C-EMO era el mejor local de los tres y obtuvo el peor LB; C-EMO-SYN, el segundo local, obtuvo el mejor).

Nota de comparabilidad: el local de v08 es el `corrected_cv` del desarrollo (1.483 filas, protocolo D-016) y no es directamente comparable con el OOF histórico de v01–v07 (1.731 filas, solo near-duplicados); el gap de v08 se calcula contra ese local corregido, igual que el +0,0059 implícito de v03 en la misma base (0,78306 − 0,7772).

### 26.5 Hallazgos del corpus en la fase v2

El análisis de los canales de cita/atribución y negación produjo una caracterización nueva del corpus sobre el desarrollo: solo el **2 %** de los tweets contiene comillas, hay **0 RT**, los marcadores de atribución (segun/dijo/afirmo y similares) aparecen en el **6,9 %** y la negación en el **25,5 %**. Hallazgo de herramienta: `es_core_news_md` **no emite** la etiqueta de dependencia `dep_=neg`; la negación se detectó combinando `advmod` con `Polarity=Neg`.

### 26.6 Hipótesis abiertas

1. **Cambio de calibración de las entradas del meta**: el meta se entrena con scores OOF de bases entrenados en 4/5 del desarrollo, pero en la sonda predice el test con bases entrenados con todo el train; ese cambio de calibración de las entradas no se midió y es una hipótesis abierta. Se suma el gap dev-LB negativo de la familia union (−0,013 a −0,016).
2. **Backlog v2 cerrado al 100 % (24 de 24 líneas)**: el paquete P8 evaluó las cinco líneas restantes: calibración Platt/isotónica anidadas (mejora Brier 0.1583 -> 0.1199 y log-loss 0.5002 -> 0.4035, sin cambio de F1 con umbrales reajustados); encoder de entidades por mención/hashtag/dominio (cobertura informativa 0.5 %, cerrada por sparsity); umbrales por rebanada con shrinkage (curva monótona, el umbral local sobreajusta, -0.002 frente al global anidado); especialistas por rebanada (11 de 21 celdas significativas tras Benjamini-Hochberg, pero rinden de media -0.045 dentro de su rebanada); y espacios de features por etiqueta (-0.0253, 0/5 pliegues). Ninguna supera la regla de adopción (D-016).

## Referencias

Las 45 referencias completas del estado del arte están en `references.bib` y `reports/estado_arte.csv`. Se listan aquí las citadas en este informe, con DOI/URL verificados:

1. Álvarez-Carmona, M. Á., et al. (2018). Overview of MEX-A3T at IberEval 2018: Authorship and Aggressiveness Analysis in Mexican Spanish Tweets. CEUR-WS Vol-2150. https://ceur-ws.org/Vol-2150/overview-mex-a3t.pdf
2. Aragón, M. E., et al. (2019). Overview of MEX-A3T at IberLEF 2019. CEUR-WS Vol-2421. https://ceur-ws.org/Vol-2421/MEX-A3T_overview.pdf
3. Arango, A., Pérez, J., y Poblete, B. (2019). Hate Speech Detection is Not as Easy as You May Think: A Closer Look at Model Validation. SIGIR 2019, 45–54. DOI 10.1145/3331184.3331262
4. Argota Vega, L. E., Reyes-Magaña, J. C., Gómez-Adorno, H., y Bel-Enguix, G. (2019). MineriaUNAM at SemEval-2019 Task 5: Detecting Hate Speech in Twitter using Multiple Features in a Combinatorial Framework. SemEval-2019, 447–452. DOI 10.18653/v1/S19-2079
5. Basile, V., Bosco, C., Fersini, E., Nozza, D., Patti, V., Rangel Pardo, F. M., Rosso, P., y Sanguinetti, M. (2019). SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter. SemEval-2019, 54–63. DOI 10.18653/v1/S19-2007
6. Bassignana, E., Basile, V., y Patti, V. (2018). HurtLex: A Multilingual Lexicon of Words to Hurt. CLiC-it 2018, CEUR-WS Vol-2253. DOI 10.4000/books.aaccademia.3085
7. Buda, M., Maki, A., y Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. Neural Networks, 106, 249–259. DOI 10.1016/j.neunet.2018.07.011
8. Davidson, T., Warmsley, D., Macy, M., y Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. ICWSM 2017, 512–515. DOI 10.1609/icwsm.v11i1.14955
9. Davidson, T., Bhattacharya, D., y Weber, I. (2019). Racial Bias in Hate Speech and Abusive Language Detection Datasets. ALW 2019, 25–35. DOI 10.18653/v1/W19-3504
10. ElSherief, M., Kulkarni, V., Nguyen, D., Wang, W. Y., y Belding, E. (2018). Hate Lingo: A Target-Based Linguistic Analysis of Hate Speech in Social Media. ICWSM 2018, 42–51. DOI 10.1609/icwsm.v12i1.15041
11. ElSherief, M., Ziems, C., Muchlinski, D., Gupta, V., Kambhatla, S., y Yang, W. (2021). Latent Hatred: A Benchmark for Understanding Implicit Hate Speech. EMNLP 2021, 345–363. DOI 10.18653/v1/2021.emnlp-main.29
12. Fortuna, P., y Nunes, S. (2018). A Survey on Automatic Detection of Hate Speech in Text. ACM Computing Surveys, 51(4), 85:1–85:30. DOI 10.1145/3232676
13. Founta, A.-M., et al. (2018). Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior. ICWSM 2018. DOI 10.1609/icwsm.v12i1.14991
14. García-Díaz, J. A., Cánovas-García, M., Colomo-Palacios, R., y Valencia-García, R. (2021). Detecting misogyny in Spanish tweets. An approach based on linguistics features and word embeddings. Future Generation Computer Systems, 114, 514–524. DOI 10.1016/j.future.2020.08.032
15. García-Díaz, J. A., Jiménez-Zafra, S. M., García-Cumbreras, M. Á., y Valencia-García, R. (2022). Evaluating feature combination strategies for hate-speech detection in Spanish using linguistic features and transformers. Complex & Intelligent Systems, 8, 4901–4916. DOI 10.1007/s40747-022-00693-x
16. Garibo i Orts, Ò. (2019). Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter at SemEval-2019 Task 5: Frequency Analysis Interpolation for Hate in Speech Detection. SemEval-2019, 460–463. DOI 10.18653/v1/S19-2081
17. Jahan, M. S., y Oussalah, M. (2023). A systematic review of hate speech automatic detection using natural language processing. Neurocomputing, 546, 126232. DOI 10.1016/j.neucom.2023.126232
18. Joachims, T. (1998). Text Categorization with Support Vector Machines: Learning with Many Relevant Features. ECML-98, 137–142. DOI 10.1007/BFb0026683
19. Lipton, Z. C., Elkan, C., y Naryanaswamy, B. (2014). Optimal Thresholding of Classifiers to Maximize F1 Measure. ECML PKDD 2014, 225–239. DOI 10.1007/978-3-662-44851-9_15
20. MacAvaney, S., Yao, H.-R., Yang, E., Russell, K., Goharian, N., y Frieder, O. (2019). Hate speech detection: Challenges and solutions. PLOS ONE, 14(8), e0221152. DOI 10.1371/journal.pone.0221152
21. Malmasi, S., y Zampieri, M. (2017). Detecting Hate Speech in Social Media. RANLP 2017, 467–472. DOI 10.26615/978-954-452-049-6_062
22. Mishra, P., Del Tredici, M., Yannakoudakis, H., y Shutova, E. (2018). Neural Character-based Composition Models for Abuse Detection. ALW 2018. DOI 10.18653/v1/W18-5101
23. Nan, Y., Chai, K. M. A., Lee, W. S., y Chieu, H. L. (2012). Optimizing F-measure: A Tale of Two Approaches. ICML 2012. DOI 10.48550/arXiv.1206.4625
24. Nina-Alcocer, V. (2019). HATERecognizer at SemEval-2019 Task 5: Using Features and Neural Networks to Face Hate Recognition. SemEval-2019, 409–415. DOI 10.18653/v1/S19-2072
25. Nobata, C., Tetreault, J., Thomas, A., Mehdad, Y., y Chang, Y. (2016). Abusive Language Detection in Online User Content. WWW 2016, 145–153. DOI 10.1145/2872427.2883062
26. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
27. Pereira-Kohatsu, J. C., Quijano-Sánchez, L., Liberatore, F., y Camacho-Collados, M. (2019). Detecting and Monitoring Hate Speech in Twitter. Sensors, 19(21), 4654. DOI 10.3390/s19214654
28. Plaza-del-Arco, F. M., Molina-González, M. D., Martin, M., y Ureña-López, L. A. (2019). SINAI at SemEval-2019 Task 5: Ensemble learning to detect hate speech against inmigrants and women in English and Spanish tweets. SemEval-2019, 476–479. DOI 10.18653/v1/S19-2084
29. Plaza-del-Arco, F. M., Molina-González, M. D., Ureña-López, L. A., y Martín-Valdivia, M. T. (2021). Comparing pre-trained language models for Spanish hate speech detection. Expert Systems with Applications, 166, 114120. DOI 10.1016/j.eswa.2020.114120
30. Plaza-del-Arco, F. M., Halterman, A., Weerasooriya, T., y Dredze, M. (2021). A Multi-Task Learning Approach to Hate Speech Detection Leveraging Sentiment Analysis. IEEE Access, 9. DOI 10.1109/ACCESS.2021.3103697
31. Poletto, F., Basile, V., Sanguinetti, M., y Bosco, C. (2021). Resources and benchmark corpora for hate speech detection: a systematic review. Language Resources and Evaluation, 55, 477–523. DOI 10.1007/s10579-020-09502-8
32. Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H., y Pierrehumbert, J. (2021). HateCheck: Functional Tests for Hate Speech Detection Models. ACL-IJCNLP 2021, 41–58. DOI 10.18653/v1/2021.acl-long.4
33. Schmidt, A., y Wiegand, M. (2017). A Survey on Hate Speech Detection using Natural Language Processing. SocialNLP 2017, 1–10. DOI 10.18653/v1/W17-1101
34. Silla, C. N., y Freitas, A. A. (2011). A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22, 31–72. DOI 10.1007/s10618-010-0175-9
35. Stamatatos, E. (2009). A survey of modern authorship attribution methods. JASIST, 60(3), 538–556. DOI 10.1002/asi.21001
36. Stamatatos, E. (2017). Authorship Attribution Using Text Distortion. EACL 2017, 1138–1149. DOI 10.18653/v1/E17-1107
37. Vidgen, B., y Derczynski, L. (2020). Directions in abusive language training data, a systematic review: Garbage in, garbage out. PLOS ONE, 15(12), e0243300. DOI 10.1371/journal.pone.0243300
38. Vidgen, B., Harris, A., Nguyen, D., Tromble, R., Hale, S., y Margetts, H. (2019). Challenges and frontiers in abusive content detection. ALW 2019, 80–93. DOI 10.18653/v1/W19-3509
39. Wiegand, M., Ruppenhofer, J., Schmidt, A., y Greenberg, C. (2018). Inducing a Lexicon of Abusive Words — a Feature-Based Approach. NAACL-HLT 2018, 1046–1056. DOI 10.18653/v1/N18-1095
40. Wiegand, M., Ruppenhofer, J., y Kleinbauer, T. (2019). Detection of Abusive Language: the Problem of Biased Datasets. NAACL-HLT 2019, 581–588. DOI 10.18653/v1/N19-1060
41. Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259. DOI 10.1016/S0893-6080(05)80023-1
42. Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., y Kumar, R. (2019). Predicting the Type and Target of Offensive Posts in Social Media. NAACL-HLT 2019, 1415–1420. DOI 10.18653/v1/N19-1144

---

### Apéndice: artefactos de evidencia citados

- Gobernanza: `PROJECT_STATE.md`, `DECISIONS.md` (D-001 a D-015), `RUNLOG.md`, `experiments.csv` (51 experimentos).
- Datos y auditoría: `reports/auditoria_resumen.json`, `reports/data_dictionary.csv`, `reports/leakage_audit.md`, `data/processed/grupos_near_duplicados.csv`.
- Estado del arte e hipótesis: `reports/estado_arte.md`, `references.bib`, `reports/hipotesis.md`.
- Análisis: `reports/eda_resumen.md` (43 artefactos en `reports/eda/`), `reports/error_analysis.md`, 26 figuras en `reports/figuras/`.
- Resultados: `reports/variantes_representaciones.csv`, `reports/features_densas_resultados.csv`, `reports/ensembles_resultados.csv`, `reports/optuna_trials.csv`, `reports/dl_desde_cero_resultados.csv`, `reports/control_pseudo_autores.json`, `reports/fase13a_refit_umbral_metricas.json`, `reports/fase13d_sondas_robustez.json`.
- Competencia: `submissions/leaderboard_log.csv`, 8 submissions en `submissions/`, reglas en `docs/competition_rules.md`.
- Auditoría final y reproducibilidad: `reports/auditoria_final.md` (veredicto GO; scripts de verificación en `/tmp/opencode/` y copia aislada en `/tmp/opencode/repro_kaggle_toxic/`).

---

## 27. Investigación posterior (fase v4 «Advanced Research v4»)

Esta sección documenta una fase de investigación posterior a la entrega, abierta
el 2026-09-25 con una especificación de 108 secciones
(`docs/advanced_research_v4_spec.md`) bajo la pregunta «qué propiedad de los
datos, de las etiquetas, de la representación o del proceso de aprendizaje
todavía no estamos explotando?». No modifica el modelo final: su valor es
acotar con evidencia qué parte del error es ruido de anotación, qué parte es
dificultad del modelo y qué parte es límite de la representación.

### 27.1 Protocolo

Se mantiene íntegro el protocolo v2 (D-016): grupos = near-duplicado ∪
pseudo-autor, desarrollo de 1.483 tweets, holdout congelado de 248, `corrected_cv`
como métrica primaria y bootstrap pareado por tweet (2.000 remuestreos). La
referencia V03 se verificó byte a byte al abrir la fase (5/5 hashes, `corrected_cv`
0.777211 reproducida exactamente) y de nuevo en la auditoría final (veredicto GO,
0 violaciones de reglas en 12 scripts). Las 26 líneas del backlog se
pre-registraron antes de experimentar para controlar la multiplicidad.

### 27.2 Resultados por track

| Track | Resultado | Cifra clave |
|---|---|---|
| A. Integridad de datos y ruido de etiqueta | Cerrado | 15 clusters de texto repetido (84 tweets, 4.85 %), solo **2 contradictorios**; el error OOF es **menor** dentro (HS 0.046 vs 0.218) |
| B. Dureza de ejemplos y reweighting | Cerrado | El hard example mining **empeora** (−0.0101 con IC negativo); bajar el peso de los que más violan el umbral **mejora** (+0.0036 en LR, +0.0086 en SVC) |
| C. Weighting de la representación | Cerrado | log-odds 0.7571, chi² 0.6622, MI 0.7378, BM25 0.7748-0.7755, binario 0.7738: **todas degradan** frente a 0.7772 |
| D. Representaciones sparse nuevas | Cerrado | Afijos 0.7758, interacciones y ventanas por debajo: el canal de morfología (brecha FASE 13) queda cerrado |
| E. Modelo conjunto de etiquetas | Cerrado | Label Powerset sobre 6 y 4 combinaciones: mejor 0.7713, **no supera** a los tres binarios independientes |
| F. Robustez y shift | Cerrado | 2 de 10 variables no supervisadas con shift; V03 da **0.7568 en textos únicos** y **0.9769 en los 68 near-duplicados**; la propagación por near-duplicado tiene techo nulo |
| G. Modelos estadísticos alternativos | Mixto | La verosimilitud class-conditional aporta **+0.0041** en fusión tardía; el **SVC RBF sobre SVD 300 llega a 0.7904** con IC frágil |
| H. Fusión y calibración del meta | Cerrado | El shift entre regímenes de entrenamiento es pequeño (KS 0.062) y **no causa daño**; se refuta la hipótesis abierta de la fase v2 |

### 27.3 Tres resultados que cambian la lectura del problema

1. **El error no es ruido de anotación.** La contradicción textual es un
   fenómeno escaso (2.8 % de los ejemplos) y concentrado en un único cluster de
   48 tweets con el mismo patrón de insulto. El modelo clasifica bien los
   ejemplos ambiguos, y las tres estrategias de la especificación (baseline,
   downweight, eliminación de contradicciones extremas) mueven la F1 en ±0.0005.
2. **La forma de medir la dureza importa, y la propuesta por la BCE no sirve en
   este corpus.** La dificultad por etiqueta ordena bien el error (r ≈ −0.80),
   pero **agregar las tres etiquetas rompe el orden** (r = −0.136) por
   compensación: los ejemplos que fallan más tienen la BCE media *menor* (0.58
   frente a 1.34). La distancia a la frontera de decisión sí es monótona
   (r = +0.345) y es la que se usó.
3. **La mejora local no es un predictor de lageneralización.** Por primera vez se
   tenía un candidato que superaba los tres criterios de adopción *y* una
   medición externa limpia: `meta36_downweight` (corrected_cv 0.7882 = +0.0110,
   4/5 pliegues, IC95 del delta pareado [+0.0054, +0.0200], corrección neta +34
   frente a V03). El holdout congelado, que no había participado en ninguna
   decisión de la fase, lo rechazó: **0.8163 frente a 0.8285 de V03 (delta
   −0.0122)**.

### 27.4 Decisión y efectos sobre la entrega

- **Ningún candidato de la fase v4 se adopta.** V03 sigue siendo el modelo y la
  submission finales; no se generó ninguna sonda nueva de leaderboard (siguen
  siendo 12) y la submission v12 se conserva como artefacto reproducible
  marcado `DESCARTADO` en `submissions/submission_manifest.csv`.
- El holdout queda agotado (5/5 evaluaciones) y su valor queda registrado en el
  registro: es la única partición que no participó en ninguna decisión y es la
  que detecta la falta de transferencia.
- Se añade a `experiments.csv` un total de **15 experimentos nuevos** (143 filas) y
  cinco decisiones (D-025 a D-029), con la misma trazabilidad que las fases
  anteriores.
- Los entregables se actualizan sin alterar lo ya validado: el notebook incorpora
  la sección 17 (44 celdas, 0 errores, 36 comprobaciones [OK]) y este informe
  incorpora la sección 27; los conteos verificados por el notebook pasan a 143
  experimentos, 39 scripts y 12 sondas.

### 27.5 Lo que la fase deja como evidencia reutilizable

- La contradicción de etiquetas se puede cuantificar de forma barata y
  automatizable (cuatro criterios de agrupamiento, 3 s de cómputo), y sirve para
  descartar de entrada toda la familia de "reparar el ruido de anotación".
- Cualquier reweighting futuro en este proyecto debe usar la distancia a la
  frontera o la violación del umbral, con signo **negativo** sobre los ejemplos
  que el modelo falla, y debe validarse con el holdout antes de cualquier sonda.
- La descomposición del desempeño por subconjunto (0.7568 en textos únicos frente
  a 0.9769 en near-duplicados) es la explicación más directa de por qué la F1
  agregada es una métrica frágil en un corpus con repetición.

## 28. Investigación posterior — Embedding From Scratch (v5 y v5.1)

Esta seccion documenta la última fase del proyecto: **aprender una representación
vectorial de las palabras desde cero**, sin transformers ni embeddings preentrenados, y
probar si aporta información que el modelo final V03 no tenga ya. Cubre dos fases:
**v5** (¿de dónde salen los datos y qué pasa al cambiar de corpus?) y **v5.1** (¿alguna
configuración de la arquitectura aporta más?).

### 28.1 Motivación y resultado en una línea

V03 funciona porque lee la **superficie** del texto: el insulto exacto, la mención, la
deformación ortográfica. Lo que no tiene es una noción de **similitud**, y con 1.483
ejemplos etiquetados no hay forma de aprenderla. La hipótesis era que una representación
distributiva añadiría esa dimensión. **El resultado es negativo y está documentado como
tal**: la representación mejora de forma medible, pero no aporta a la tarea, y **V03 se
mantiene** en `corrected_cv` **0.777211** (LB **0.78306**).

### 28.2 Corpus: construcción y autorización

| corpus | tokens | documentos | licencia | papel |
|---|---|---|---|---|
| Wikipedia ES 5M | 5.011.404 | 1.694 | CC BY-SA 4.0 | tramo inferior de la curva |
| Wikipedia ES 10M | 10.002.400 | 2.847 | CC BY-SA 4.0 | corpus de referencia |
| Wikipedia ES 50M | 50.000.028 | 21.529 | CC BY-SA 4.0 | tramo superior |
| mC4-es 10M | 10.000.239 | 16.738 | ODC-BY 1.0 | registro web |
| mixto 20M | 20.000.045 | — | unión de los dos anteriores | efecto del registro |

Pipeline de siete pasos, **idéntico para todos** para que la comparación sea de cantidad de
datos y no de receta: `download` → `raw` → `clean` → `dedup` → `normalize` → `tokenize` →
`shards` + `manifest`. La normalización es la variante **EMB-NORMAL**: minúsculas
conservando tildes, URLs, menciones, hashtags y números reducidos a marcadores estables,
**sin corrector ortográfico y sin reconstrucción de tildes**.

La **autorización** (D-030) permite corpus externos **sin etiquetar solo para aprender la
representación** y prohíbe todo uso supervisado. Se registra tal como se comunicó, sin
documentación escrita inventada. OSCAR, que era *gated*, no se forzó.

**Detalle de trazabilidad que resultó ser una corrección**: el manifiesto guardaba el
`sha256` del **volcado**, que no identifica la escala porque 5M, 10M y 50M de Wikipedia
salen del mismo tramo. La identidad de un entrenamiento usa el `sha256` del corpus **ya
procesado**, que sí distingue las tres escalas (D-035).

### 28.3 Cobertura y OOV

La cobertura se mide contra el vocabulario TOXIC de train y test (uso como diagnóstico,
autorizado; no como señal).

| corpus | por tipo | por token |
|---|---|---|
| Wikipedia 5M | 0.6302 | 0.8646 |
| Wikipedia 10M | 0.6839 | 0.8833 |
| Wikipedia 50M | 0.7703 | 0.9238 |
| web mC4-es 10M | 0.8256 | 0.9512 |
| mixto 20M | 0.8413 | 0.9553 |

**Hallazgo central de la fase**: la cobertura y la aportación **se mueven por separado**. El
corpus web sube la cobertura del 0,68 al 0,83 y la aportación a V03 cambia en milésimas. Y la
hipótesis de que faltara señal por "no tener tildes" queda refutada: el 8,5 % de los tipos
ausentes sí lo están, y el 66,6 % son **deformaciones con pérdida de vocales**.

### 28.4 FastText y el control Word2Vec

El control es el mismo binario con las subpalabras apagadas, que es Word2Vec. La
comparación aísla **una sola cosa**:

| modelo | tipos con vector de 6.059 | fusión |
|---|---|---|
| FastText 200d (subpalabras 3-6) | **6.059** | 0.7818 |
| Word2Vec-control (10M) | 5.134 | 0.7762 |
| FastText 200d (50M) | **6.059** | 0.7803 |
| Word2Vec-control (50M) | 4.412 | 0.7773 |

Los 925 tipos que el control pierde son formas ortográficas, y son las que deciden la
etiqueta en un corpus de tuits. La ventaja de las subpalabras tiene forma de **joroba** con
máximo en 10M, no una relación monótona con la escala (D-034).

### 28.5 Downstream y la regla de escala

| pooling | corrected_cv |
|---|---|
| media | 0.6603 |
| media + máximo | 0.6388 |
| ponderado por IDF | 0.6348 |
| **suelo aleatorio** | **0.5404** |

La regla de escala importa más que el pooling: al fusionar, el bloque denso se normaliza
**L2 por documento**. Estandarizarlo hace que las 200 dimensiones dominen al bloque disperso
y la fusión **pierda 0,09 de F1** (0.7818 con L2 frente a 0.6901) sin que eso diga nada
sobre la representación: son los mismos datos.

### 28.6 Fusión y escala del corpus

Cinco configuraciones, y el delta de aportación vive siempre entre **+0.0001 y +0.0046**,
con el intervalo de bootstrap tocando el 0:

| corpus | fusión | delta |
|---|---|---|
| Wikipedia 5M | 0.7757 | **−0.0015** |
| Wikipedia 10M | 0.7818 | +0.0046 |
| Wikipedia 50M | 0.7803 | +0.0031 |
| web mC4-es 10M | 0.7812 | +0.0040 |
| mixto 20M | 0.7776 | +0.0004 |

**Conclusión de la curva**: multiplicar por diez los tokens (5M → 50M) **reduce** el delta.
El techo no es la representación, y **no se escala el corpus**: tres de las cinco condiciones
de la spec fallan, y las que fallan son de evidencia, no de coste (D-041).

### 28.7 Arquitectura (v5.1): las cinco familias

Todas sobre Wikipedia 10M, cambiando **un parámetro** cada vez. Criterio de adopción de dos
niveles: **delta ≥ +0,005 Y el intervalo de confianza no incluye el 0** (D-036).

| familia | mejor fusión | delta | IC95 | veredicto |
|---|---|---|---|---|
| vector_size | 0.7823 (100d) | +0.0050 | [−0.0028, +0.0130] | cerrada (inerte) |
| window | 0.7830 (ws 3) | +0.0057 | [−0.0021, +0.0136] | cerrada (inerte) |
| min_count | 0.7818 (baseline) | +0.0046 | [−0.0032, +0.0124] | cerrada (inerte) |
| subwords | 0.7818 (baseline) | +0.0046 | [−0.0032, +0.0124] | cerrada (inerte) |
| negative | 0.7818 (baseline) | +0.0046 | [−0.0032, +0.0124] | cerrada (inerte) |

**14 configuraciones, 0 adoptables.** Tres lecturas que quedan como resultado:

- **La dimensión no es la palanca**: con 1.483 ejemplos, 100, 200 y 300 distancias dan lo
  mismo (rango 0.0017).
- **Menos ventana rinde más** (3 > 5 > 7), y el intervalo de subpalabras se queda en el
  punto medio 3-6: acortar o alargar empeora. La señal útil es **local y ortográfica**.
- **`min_count` no puede funcionar con subpalabras**: bajar el umbral agranda el vocabulario
  general pero no añade ni un tipo de TOXIC con vector, porque fastText ya compone vector
  para cualquier tipo consultation. Su vía de mejora ya está ocupada.

### 28.8 Complementariedad y resultados negativos

- **Qué arregla**: el embedding aporta **la misma información que V03 con más ruido**. Su
  variante con mejor delta (WS-3) no arregla más filas que el baseline (8 frente a 8):
  arregla lo mismo y **rompe menos** (2 frente a 5). Toda la ventaja está en el *damage*.
- **Rangos**: el embedding aislado se mueve en 0,0297 entre 0,6397 y 0,6694; la fusión se
  mueve en 0,0063. El rango de la representación es **4,7 veces mayor** que el de la tarea.
- **Resultados negativos explícitos**: 100d y 300d (no superan el baseline), ventanas 3 y 7
  (mejoran el embedding pero no la fusión), min_count 2 y 5 (ninguno supera el 3), subpalabras
  3-5 y 3-7 (ninguno supera el 3-6), negative 5 (cae por debajo del 10), mezcla de corpus
  (la peor: +0.0004), estandarizar el bloque denso (−0.0871), los cinco corpora por debajo
  del gate, y el meta-modelo anidado de v5 (−0.0094).

### 28.9 Contabilidad de la fase

- 22 scripts de la línea del embedding y 138
  evaluaciones downstream registradas, con doble contabilidad (entrenamientos ≠ evaluaciones).
- Presupuesto: **9 de 9**
  exploratorios y **0 de 1** de
  reserva. La reserva queda **intacta** porque su único fin era confirmar una mejora ya
  vista, y no la hay.
- 42 decisiones registradas en total; esta línea ocupa D-030 a D-042.

### 28.10 Conclusión

La fase entrega dos conclusiones separadas y las mantiene separadas a propósito, porque en
v5 se confundieron una vez:

1. **Representación**: la arquitectura **sí** cambia la calidad del embedding (rango
   0,0297) y su forma revela que la señal útil es local y ortográfica. Es un resultado
   positivo como propiedad del embedding.
2. **Tarea**: ninguna configuración aporta a V03 por encima del ruido. El mejor delta de la
   tabla es +0,0057 con IC95 [−0,0021, +0,0136]. Es un resultado negativo, y ninguno anula
   al otro.

**V03 sigue siendo el modelo final.** La única variable que la evidencia deja en pie es el
**tamaño de los datos etiquetados** (1.483): con más etiquetas, tanto el bloque disperso
como el denso podrían aportar más de lo que aportan hoy.

*Sección generada por `experiments/embedding_21_documentacion_v5_1.py` (EBD-27 (P9)) el
2026-09-26. Sin push: la rama sigue local.*


---

## Addendum P45 (2026-09-27) - La metrica, verificada; la estructura conjunta, cerrada

Este addendum responde a las dos preguntas que quedaban abiertas antes de abrir cualquier
otra familia de modelos.

### 1. La metrica local es exactamente la que se optimizo, y queda demonstrated

`corrected_cv = 0.7772114525165721` se reproduce con diferencia `+0.000e+00` desde el OOF
canonico. La magnitud del proyecto, `f1_media_columnas`, es **identica** a
`sklearn.metrics.f1_score(..., average="macro")` sobre entrada multilabel: diferencia
`0.000e+00` y **0 discrepancias en 600 casos aleatorios**.

Un detalle que conviene no pasar por alto: la media de los cinco pliegues da `0.7773117212`
y el global sobre el OOF da `0.7772114525`. No son la misma cifra. El proyecto reporta el
global, que es el correcto, pero cualquier lectura que promovie las dos como intercambiables
introduce un error de `+0.0001002686`.

El parametro `average` que aplica el host no es legible. Eso motivo un discriminante de
escala: `samples` da `0.334` sobre el OOF de V03, mientras el host reporto `0.78306` para la
submission de V03. Una brecha de `0.449` no se explica por desplazamiento de prevalencia, de
modo que **`samples` queda excluido**. Las tres hipotesis restantes (`macro`, `micro`,
`weighted`) estan a menos de `0.0036` entre si y ordenan los candidatos de umbrales igual
(spearman `>= 0.98` sobre 1000 configuraciones).

**`METRIC_MATCH = TRUE`**, con la salvedad escrita: esto demuestra la igualdad con `macro`, no
una lectura directa del host.

### 2. La estructura conjunta existe, y no se puede explotar

`LABEL_STATE = 4*HS + 2*TR + AG`. Los estados `010` y `011` **nunca se observan**:
`TR=1` implica `HS=1` en los 265 casos de train con `TR=1`. Es una restriccion estructural
fuerte y exacta.

Y hay estructura de error conjunto, de verdad. El contraste de independencia de los tres
indicadores de error, dado el estado verdadero, da **chi2 343.156 con 20 grados de libertad,
p < 1e-6**, con desviaciones de hasta 116 casos en una sola celda. Los errores de HS, TR y AG
**no** son independientes.

La pregunta es si eso se traduce en senal. No. La extraccion mas directa, la correccion
estructural dura, es **negativa en las dos formas**: `-0.000646` al forzar `TR=0` si `HS=0`, y
`-0.003542` al forzar `AG=0` si `HS=0`. Y las formulaciones de aprendizaje conjunto ya estan
todas en el registro con resultado negativo o sin transferencia: Label Powerset
(`V4-E01-E03`, 12 variantes, todas bajo V03), classifier chains en ambos ordenes
(`E2-01-HS-TR-AG` 0.773674 y `E2-01-HS-AG-TR` 0.774429, ambos bajo 0.777211), espacio
separado por etiqueta, acoplamiento suave entre cabeceras, meta-modelos.

**`JOINT_LABEL_STRUCTURE = CLOSED`.** No se implementa ningun decoder estructurado: su unica
diferencia respecto a la correccion dura es el peso, y la version de peso infinito ya es
negativa.

### 3. Lo que cambia y lo que no

No cambia el campeon. **V03 sigue con `corrected_cv = 0.7772114525165721`, LB 0.78306,
sha256 `7d7478f2`, intacta.** 0 submissions en esta fase; las 3 reservadas siguen reservadas.

Lo que si cambia es la confianza en las 311 filas de registro que hay detras de esa cifra: la
base de comparacion esta verificada, asi que cualquier delta futuro es interpretable.

### 4. Advertencia sobre la transferencia

El pearson entre `local_f1` y `kaggle_f1` es **0.4876**, no alto. La transferencia
local->leaderboard es ruidosa, y es coherente con el hecho historico de que **0 de 22
submissions superaron a V03** pese a metricas locales altas. Conviene tenerlo presente antes de
interpretar cualquier mejora local pequena como promesa de mejora en el externo.

*Seccion generada por `experiments/v45_00_metrica_audit.py` y
`experiments/v45_01_joint_label_structure.py` el 2026-09-27. Sin push: la rama sigue local.*


---

## Addendum P46 (2026-09-28) - El campeon verificado

### El veredicto

```text
leaderboard_max_verified = 0.78306
champion                = v03_lr_union_thr  (V03)
sha256                  = 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
CASE                    = B
CHAMPION_STATUS         = INCONSISTENTE_AUTOCORREGIDO
```

**Ninguna de las 22 submissions enviadas supero a V03.** Cada una tiene artefacto en disco con
su sha256, y el registro no tiene duplicados, nulos ni valores fuera de rango.

### Por que la contradiccion no era una contradiccion

`CHAMPION.md` atribuia a `v16_ch1` un LB de 0.78535 y a `v16_ch5` un LB de 0.78650. La columna
que contiene esos numeros se titula, en su propio encabezado, **`LB combinado`**, y unas lineas
mas abajo el fichero declara la formula que la produce:

```text
LB combinado = 0.78306 + 0.12011 x delta_tasa + aciertos_netos_oof x 2.804e-05
```

Recalculada con los parametros que el propio texto da, esa formula devuelve 0.785342 y
0.786495: desviaciones de **7.9e-06** y **5.2e-06** respecto a los valores afirmados. Eran
**predicciones** del modelo causal de la fase v16, no puntuaciones del host.

Y lo mas importante: las mediciones reales de esas mismas submissions estan **en el mismo
fichero**, en la seccion *Estado al cierre de V16*, y son **0.77828** y **0.77495**, ambas
marcadas REJECTED. No habia dos fuentes en conflicto. Habia una prediccion y una medicion, y la
medicion es la que vale. El propio CHAMPION.md ya lo habia escrito, en una tabla 110 lineas mas
abajo de la que se estaba citando.

Se conserva la tabla original intacta como registro de lo que se predijo, con un aviso que
marca la columna como prediccion. No se borra historia.

### Las cuatro magnitudes que no deben confundirse

| magnitud | valor | quien |
|---|---:|---|
| mejor corrected_cv utilizable | 0.7772114525165721 | OOF canonico de 1483 filas |
| mejor local del log | 0.80390 | `v05_optuna`, LB 0.77602 |
| mejor submission por LB | **0.78306** | `v03_lr_union_thr` |
| mejor modelo congelado | 0.7772114525 | V03, LB 0.78306 |

El mejor local no es el mejor LB. `v05_optuna` tiene el mejor local del registro y pierde
0.007 de LB. Esa distancia es la razon por la que el log de leaderboard, y no la mejor cifra
local, debe decidir el campeon.

### Un defecto del registro que conviene conocer

La columna `macro_f1` de `experiments.csv` **mezcla tres bases de medida**: 4 filas sobre el
holdout2 de 248 filas, 46 sobre el split D-004 que D-015/D-016 supersedieron, y el resto sin
base declarada. Las filas historicas de V03 registran 0.8322 y 0.8356, no 0.7772114525.

Un maximo ingenuo sobre esa columna devuelve **S08 con 0.8743**, que es una medida de holdout.
Cualquiera que ordene el registro por esa columna sin clasificar la base obtiene rankings
invalidos. La magnitud de CV utilizable se lee del artefacto congelado
`features/oof_v03_final.npz`.

No se reescribieron las filas historicas. Cambiar 46 valores seria reescribir historia sin
evidencia de que el valor corregido sea el que se midio en su momento.

### Correccion a P45

`METRIC_MATCH = TRUE` era demasiado fuerte. Queda desdoblado:

```text
LOCAL_METRIC_REPRODUCED           = TRUE
KAGGLE_SCORER_DIRECTLY_VERIFIED   = FALSE
```

El codigo del scorer del host no esta en el repositorio. Lo que si esta demostrado es que la
metrica local es exactamente `f1_score(average="macro")` y que `corrected_cv` se reproduce al
ultimo digito. Eso basta para que las comparaciones internas del proyecto sean coherentes, y
no basta para afirmar que se ha leido el scorer de Kaggle.

Ademas, las decisiones que P45 escribio como D-030 y D-031 ocupan IDs ya usados por la fase
v5. Renumeradas a D-115 y D-116.

*Seccion generada por `experiments/v46_00_champion_audit.py` y
`experiments/v46_01_integrar_notebook.py`. Sin push: la rama sigue local.*


---

## Addendum P47 (2026-09-28) - No queda via de datos

### El veredicto

```text
CASE               = A + D
DATA_EXPANSION     = CLOSED
TRAIN_AUGMENTATION = CLOSED
```

### El train esta completo

```text
train oficial etiquetado   1731
  de las que en dev         1483   (usadas por V03)
  de las que en holdout     248   (agotadas)
  sin asignar                 0
test oficial               1154   (sin labels)
```

1731 = 1483 + 248, exactamente. **No queda ni una fila etiquetada oficial sin usar.** La
recomendacion de "mas datos etiquetados" que P46 dejo como unica variable viable **no es
operacionalmente posible** con este corpus. No es que resulte dificil: es que no hay nada que
usar.

### Dos cifras de la documentacion eran falsas

**519 no era el holdout.** Es el numero de positivos TR en train. El holdout tiene **248** filas.
La cifra circulaba como si fuera el tamano del holdout, y por eso la mencionaba como "519 filas
de holdout" al recomendar mas datos.

**No existen ~44k filas sin etiquetar.** Los corpus externos existen y son grandes, pero se
miden en **tokens**: 5M, 10M, 10M, 20M y 50M, para 42808 documentos y 95.114.116 tokens en
total. Ningun dataset del proyecto tiene 44k filas. El "44" que circulaba en el repositorio son
las **44 features** del meta-modelo `E2-02-META`.

### Lo que si hay, y por que no sirve

| fuente | volumen | legal para |
|---|---|---|
| wikipedia_es (CC BY-SA 4.0) | 10M / 50M / 5M tokens | solo embeddings |
| m_c4_es (ODC-BY 1.0) | 10M tokens | solo embeddings |
| mezcla de ambos | 20M tokens | solo embeddings |
| 19 modelos fastText entrenados | artefactos | nada, son salidas |

D-030 autoriza los corpus externos **exclusivamente para aprender la representacion
distribuida**, y prohibe de forma explicita el entrenamiento supervisado, la generacion de
labels, el pseudo-labeling y el ajuste de thresholds. Como la familia de embeddings esta
cerrada, los 95 millones de tokens no tienen uso disponible.

### La augmentation ya existia

La directiva la abria solo si se confirmaba que no se habia ejecutado. Se habia ejecutado:
**V6-08**.

- Transformaciones: `espaciado`, `elongacion`, `repeticion_puntuacion`. Son exactamente los
  tres candidatos ortograficos que la directiva listaba.
- Colocacion: dentro del training fold, que es lo que la directiva exigia.
- Grupos: preservados. El control J0 reproduce V03 exactamente.
- Resultado: J1 (una copia por fila) = **-0.0061**. J2 (dos copias) tambien negativo.

No se creo ninguna fila experimental ficticia, y no se repitio el experimento bajo otra
denominacion.

### Donde queda el proyecto

El campeon no se mueve: V03, `corrected_cv = 0.7772114525165721`, LB 0.78306. Y la via de los
datos se cierra por evidencia.

Eso deja el proyecto **sin ninguna via abierta**: 15 familias de modelos cerradas, la
estructura conjunta cerrada, la expansion de datos cerrada, la augmentation cerrada, la
metrica verificada y el campeon verificado. Cualquier hipotesis futura tiene que salir de una
familia de modelo nueva, no de una fuente de datos nueva, porque no queda ninguna.

*Seccion generada por `experiments/v47_00_auditoria_datos.py` y
`experiments/v47_01_integrar_notebook.py`. Sin push: la rama sigue local.*


---

## Addendum P48 (2026-09-28) - El EDA se exploto por completo, y no queda senal

### La pregunta

P48 no era "que modelo probamos ahora", sino si seguimos sin mejorar porque ya no existe senal
o porque nunca convertimos el EDA en mecanismos. Son dos diagnosticos opuestos y obligan a
respuestas opuestas.

### Primera respuesta: la conversion fue completa

| magnitud | valor |
|---|---:|
| hallazgos del EDA auditados | 15 |
| con mecanismo probado y cerrado | **14** |
| sin un solo experimento | **0** |
| abiertos | 1 |

El unico abierto es la rareza de etiqueta, con un unico experimento. Y el propio informe EDA de
origen solo declaraba hipotesis explicita en 3 de sus 6 hallazgos clave: los otros tres se
convirtieron en experimentos por otras vias, pero lo hicieron.

Aplicar un criterio mas severo **empeora** el balance, no lo mejora. Exigir que el tema no tenga
ninguna fila con un veredicto de apertura, siendo literales, daria 14 de 15 igual. La conclusion
no depende del criterio.

### Segunda respuesta: la senal tampoco esta

De las 757 celdas de error de V03, **715 (94,45 por ciento)** las acierta al menos un canal OOF
historico, y **42 (5,55 por ciento) ninguna**. La tasa de error global es 0,2974 y la de
rescate 0,9445.

Aqui esta el hallazgo que decide la fase, y es contraintuitivo. Hay slices con tasa de error
claramente superior a la media:

| patron | lift de error | tasa de rescate | enriquecimiento |
|---|---:|---:|---:|
| cita | 1,60 | 0,9655 | 1,022 |
| colectivo | 1,39 | 0,9267 | 0,981 |
| negacion | 1,28 | 0,8776 | 0,929 |
| longitud larga | 1,26 | 0,9017 | 0,955 |
| hashtags | 1,21 | 0,9808 | 1,038 |

Todas con p por debajo de 0,01. **Y ninguna se enriquece en rescate.** El enriquecimiento es el
cociente entre la tasa de rescate del patron y la global, y todos rondan 1,0 o estan por debajo.
Varios son significativamente **mas bajos** que la media: colectivo en falso negativo de HS 0,882,
negacion 0,929.

Eso es la prueba de que **tener mas errores no es tener senal oculta**. Un patron donde V03 falla
mas de lo normal, y donde el resto de modelos acierta en la misma proporcion que en cualquier otro
sitio, no es una oportunidad: es una descripcion de donde V03 falla, que ya sabiamos.

Un criterio de hipotesis mal construido habria abierto aqui cinco lineas de investigacion. Todas
serian duplicados de familias ya cerradas, porque colectivo, negacion, longitud, elongacion,
hashtags y cita ya tienen 11 a 46 experimentos cada uno.

### El margen, que si tiene senal

La tasa de rescate cae de forma monotona con el margen de V03:

| bin de margen | errores | tasa de rescate |
|---|---:|---:|
| [0, 0,05) | 141 | 1,0000 |
| [0,05, 0,10) | 141 | 0,9858 |
| [0,10, 0,20) | 207 | 0,9807 |
| [0,20, 0,30) | 134 | 0,9179 |
| [0,30, 1] | 134 | 0,8134 |

Y el **85,7 por ciento de las celdas irrecuperables estan en margen alto** (36 de 42). Es la firma
de un hueco de representacion: V03 se equivoca con conviccion exactamente donde ningun canal
historico acierta.

Es un hallazgo real y con mecanismo. Pero son **42 celdas de 757**, un 5,55 por ciento del
error. Para que una hipotesis de este tipo fuera accionable haria falta que ese subconjunto
fuera grande, y no lo es.

### Decision

**`EDA_RESEARCH_GAP = NONE`.** Cero hipotesis, cero experimentos de un presupuesto de tres.

### Lo que queda abierto, que no es una hipotesis

La limitacion real de esta fase no es de senal, es de **artefacto**: los tensores OOF por canal
(`features/v2_oof_*.npz`) ya no estan en disco. Solo sobrevive la fraccion de fallo por celda, de
modo que la categoria C es una cota superior del rescate, no la identificacion del modelo que
rescata. Con esos tensores, la misma pregunta se responderia con evidencia mas fuerte. No es una
via de mejora del modelo: es una tarea de mantenimiento del repositorio, y es la unica accion
que esta fase deja sobre la mesa.

*Seccion generada por `experiments/v48_00_eda_traceability.py`,
`experiments/v48_01_residual_atlas.py` y `experiments/v48_02_integrar_notebook.py`.
Sin push: la rama sigue local.*
