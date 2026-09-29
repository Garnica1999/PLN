# Informe final  -  Clasificacion multi-etiqueta de discurso de odio en tuits en espanol (HS / TR / AG)

**Proyecto**: KAGGLE_TOXIC_2026_PUJ  -  competencia academica `tweet-toxic-2026-puj` (Kaggle id 167384, organizacion JarvisLab_PUJ).
**Curso**: Procesamiento del Lenguaje Natural, Maestria en Inteligencia Artificial, Pontificia Universidad Javeriana.
**Fecha de cierre del informe**: 2026-09-24. **Deadline de la competencia**: 2026-10-19 23:00 UTC.
**Documento hermano**: `notebooks/10_final_pipeline.ipynb` (31 celdas: 16 de codigo, todas ejecutadas, y 15 de narrativa), entregable que concentra el 30 % de la nota.

---

## Resumen ejecutivo

El proyecto aborda la clasificacion multi-etiqueta de tres columnas binarias  - HS (discurso de odio), TR (dirigido) y AG (agresividad) -  sobre 1.731 tuits de entrenamiento y 1.154 de prueba, con la restriccion explicita de no usar transformers, LLM ni embeddings preentrenados. La metrica oficial se identifico empiricamente como **F1 macro sobre las 3 columnas** mediante una sonda de submission (plantilla todo-1: LB 0.51494). El modelo final es una **union TF-IDF word(1,2) + char_wb(3,5)** (27.431 features) con **LogisticRegression balanceada C=2.0** por etiqueta y **umbrales por etiqueta 0.44 / 0.51 / 0.46**, con **LB 0.78306** (mejor de 8 envios) y OOF local 0.7989. Los resultados negativos son parte central de la evidencia: las 10 variantes de preprocesamiento (mejor +0.0035), el post-proceso jerarquico (-0.0056), los bloques densos (-0.009 a -0.032), los ensembles (mejor +0.0023 anidado), la optimizacion Optuna (OOF 0.8039 -> LB 0.77602), el control de capacidad (gap train-OOF no predice el gap local-LB) y el deep learning desde cero (mejor MLP 0.7949 OOF) no superaron la barrera de adopcion pre-registrada. Se documenta un **plateau** del leaderboard: el rango 0.77034-0.78306 equivale a **0.99 errores estandar** (n=1.154). La auditoria final emitio veredicto **GO** con condiciones, hoy satisfechas: pipeline sin fuga, reglas cumplidas y resultados reproducibles bit a bit.

Una investigacion posterior (fase v2, 2026-09-25) evaluo 24 lineas de investigacion bajo un protocolo de validacion corregido (grupos fusionados near-duplicado UNION pseudo-autor y holdout congelado). Su mejor candidato  - un meta-modelo de dependencia de etiquetas -  mejoro en desarrollo (+0,0081) y en el holdout congelado (+0,0180 pareado), pero **no transfirio al test** (LB 0,78163 frente a 0,78306 de V03); el modelo final entregado sigue siendo **V03** (seccion 26).

---

## 1. Definicion del problema

La competencia plantea una tarea de clasificacion de texto multietiqueta: dado el campo `text` de un tuit o comentario en espanol, predecir tres etiquetas binarias simultaneas:

- **HS = 1**: el texto contiene discurso de odio.
- **TR = 1**: el discurso esta dirigido contra una persona o grupo.
- **AG = 1**: el texto es agresivo.

No se trata de categorias mutuamente excluyentes (no es un problema multiclase): las tres salidas se predicen en paralelo y la metrica oficial (F1 Score del leaderboard privado) se calcula sobre las tres columnas. El enunciado oficial pone el enfasis en "el procesamiento y representacion del texto para que los modelos clasicos puedan capturar mejor las senales de toxicidad" (`docs/competition_rules.md`).

La estructura observada de las etiquetas es fuertemente condicionada: en los 1.731 ejemplos de entrenamiento, **TR=1 ocurre unicamente cuando HS=1** (0 violaciones), mientras que AG=1 con HS=0 aparece en solo 26 casos (1,5 %). Esto justifico la decision D-001 de modelar tres clasificadores binarios acoplados (formulacion multi-etiqueta) en lugar de una unica clase con seis combinaciones, y de evaluar  - y finalmente descartar -  un post-proceso jerarquico `HS=0 => TR=0, AG=0`. La estructura condicionada es coherente con el esquema de anotacion HatEval (Basile et al., 2019, DOI 10.18653/v1/S19-2007), donde TR y AG solo se anotan sobre textos que ya contienen odio.

El problema tiene tres dificultades especificas del corpus: (i) los textos son cortos (6-306 caracteres, mediana 91); (ii) el corpus esta normalizado a ASCII, sin tildes ni ene, por lo que se pierde parte de la senal morfologica del espanol; y (iii) la frontera entre "insulto interpersonal en contexto de broma o afiliacion" y "discurso de odio" es en buena parte extralinguistica, como demostro el analisis de error (seccion 20).

## 2. Restricciones de la competencia

Las reglas fueron capturadas verbatim en `docs/competition_rules.md`. Las restricciones que condicionaron todas las decisiones tecnicas son:

| Restriccion | Origen | Implicacion |
|---|---|---|
| Prohibido usar transformers o LLM (BERT, RoBERTa, etc.) | Indicacion del profesor (regla adicional) | Se descarta toda la familia de modelos preentrenados, pese a que Plaza-del-Arco et al. (2021, DOI 10.1016/j.eswa.2020.114120) muestran su techo superior en espanol |
| Prohibido usar embeddings preentrenados | "Todo desde cero" | spaCy se admite solo como anotador linguistico (POS, lemas), nunca como vector (D-003) |
| Vectorizadores entrenados desde cero | Regla del profesor | TF-IDF, conteos y features inducidas solo con datos de entrenamiento |
| El test no puede usarse para entrenamiento | Reglas de la competencia | El test solo se usa con `transform`/`predict`; jamas para ajustar umbrales o seleccionar modelos |
| Maximo 10 envios por dia; solo cuenta la mejor puntuacion | Reglas generales | Se priorizo no sobreajustar el leaderboard (D-014) |
| Formato de submission `id,HS,TR,AG` (1.154 filas) | Plantilla oficial | Validacion programatica de cada submission contra la plantilla |
| Evaluacion: 70 % leaderboard privado + 30 % notebook | Reglas academicas | El esfuerzo final se redirigio a notebook e informe (D-014) |

Restricciones autoimpuestas por integridad academica y metodologica:

- **D-002**: prohibicion absoluta de consultar o usar etiquetas o corpus externos (los textos originales de HatEval circulan publicamente; usarlos violaria la validez del resultado).
- **D-007**: cualquier lexicon externo debia registrarse en un catalogo con licencia y finalidad, y mantener siempre una variante autocontenida. Al final no se uso ningun lexicon externo.
- **D-004/D-015**: validacion agrupada por near-duplicados y por pseudo-autor.
- **Regla de decision pre-registrada**: adoptar una mejora solo con >= +0.005 absoluto en la metrica primaria y mejora en >= 3 de 5 pliegues; en la familia word+char, exigir >= +0.010 local antes de gastar una sonda de leaderboard (D-012).

La auditoria final (`reports/auditoria_final.md`) verifico el cumplimiento de cada regla con evidencia ejecutable: busqueda de `transformers|bert|gpt|llama|...` en todo el codigo (0 usos reales), ausencia de `.vector`/`.tensor`/`similarity()` de spaCy, vocabulario del artefacto final identico a un ajuste solo con train, y 8/8 submissions con formato valido. Veredicto: **APROBADO**.

## 3. Estado del arte

La fase de investigacion produjo **45 referencias verificadas** (`reports/estado_arte.md`, `reports/estado_arte.csv`, `references.bib`), con DOI resueltos en Crossref, BibTeX de ACL Anthology, PDFs de ACL/CEUR-WS y comprobaciones puntuales en OpenAlex. Cuatro DOIs fueron validados manualmente contra Crossref (MineriaUNAM, HATERecognizer, SINAI y el articulo de comparacion de transformers en espanol).

Los hallazgos que estructuraron el proyecto:

| Trabajo | Aporte | Uso en el proyecto |
|---|---|---|
| Argota Vega et al. (2019), MineriaUNAM, DOI 10.18653/v1/S19-2079 | Gano la Subtask A en espanol de HatEval **sin redes ni embeddings preentrenados**: SVM lineal + char n-gramas, n-gramas de palabras funcionales 2-4, n-gramas de puntuacion 2-4, POS, patrones morfologicos y series de palabras agresivas (macro-F1 0.73 vs baseline 0.701) | Techo de referencia de un sistema clasico; justifica la ingenieria de representacion y el clasificador lineal |
| Nina-Alcocer (2019), DOI 10.18653/v1/S19-2072 | `TFIDF+STRUCT` mejor F1 (0.780); `TFIDF+EMOTIONS` mejor EMR (0.711 vs 0.629); **combinar tres bloques degrado** el rendimiento | Alerta contra acumular features sin ablacion; motivo la aproximacion conservadora de un solo bloque de representacion |
| Plaza-del-Arco et al. (2019), SINAI, DOI 10.18653/v1/S19-2084 | La votacion de ensembles (0.707) apenas supera al SVM lineal (0.701) | Evidencia en contra de esperar mejoras automaticas por ensamblar; H10 se planteo como stacking con OOF |
| Basile et al. (2019), HatEval, DOI 10.18653/v1/S19-2007 | TR y AG no son independientes: se anotan condicionadas a HS | Soporte estructural de D-001 y de la hipotesis H8 |
| Zampieri et al. (2019), OffensEval, DOI 10.18653/v1/N19-1144 | El target es una tarea estructuralmente distinta y mas dificil que la deteccion de ofensa | Modelar TR como cabeza separada |
| ElSherief et al. (2021), Latent Hatred, DOI 10.18653/v1/2021.emnlp-main.29 | El odio implicito es un fenomeno diferenciado y dificil | Base conceptual del patron de falsos negativos de AG/HS sin insulto lexico |
| Malmasi y Zampieri (2017), DOI 10.26615/978-954-452-049-6_062; Mishra et al. (2018), DOI 10.18653/v1/W18-5101 | N-gramas de caracter para lenguaje abusivo | Justificacion de la representacion `char_wb` |
| Lipton et al. (2014), DOI 10.1007/978-3-662-44851-9_15; Nan et al. (2012), DOI 10.48550/arXiv.1206.4625 | El umbral optimo para maximizar F1 no es 0.5 | Base de la familia de experimentos de umbral (TH-01, D-012) |
| Silla y Freitas (2011), DOI 10.1007/s10618-010-0175-9 | Clasificacion jerarquica | Referencia para el post-proceso HS=0 => TR=0, AG=0 (H8) |
| Wolpert (1992), DOI 10.1016/S0893-6080(05)80023-1 | Stacked generalization | Diseno de los ensembles con evaluacion anidada |
| Wiegand et al. (2019), DOI 10.18653/v1/N19-1060; Davidson et al. (2019), DOI 10.18653/v1/W19-3504; Arango et al. (2019), DOI 10.1145/3331184.3331262 | Sesgo de datasets, variabilidad y fragilidad de la validacion en deteccion de odio | Advertencias incorporadas a las limitaciones y al control de pseudo-autores (D-015) |

El balance de compatibilidad con las restricciones fue 37/45 metodos compatibles de forma directa y 8/45 con advertencia (parcial o prohibido por depender de transformers/embeddings). La regla de decision pre-registrada (>= +0.005 y >= 3/5 pliegues) y la secuencia de experimentos E01-E10 quedaron documentadas en `reports/hipotesis.md` antes de modelar.

## 4. Dataset

Los tres archivos oficiales se descargaron con la API de Kaggle (`src/data.py:download_competition`) y se auditaron en la Fase 0 (`experiments/00_auditoria_datos.py`, `reports/auditoria_resumen.json`, `reports/data_dictionary.csv`).

| Propiedad | Valor verificado |
|---|---|
| `toxic_train.csv` | 1.731 filas; UTF-8; separador `;`; quoting CSV estandar (el quoting `QUOTE_NONE` rompe 13 filas); 205.417 bytes |
| `toxic_test.csv` | 1.154 filas; **cp1252** (6 bytes `0xBF`); solo `id,text`; 127.006 bytes |
| `submission.csv` | Plantilla `id,HS,TR,AG` separada por coma; ids identicos y en el mismo orden que el test |
| ids | Rango 20001-24999, unicos, disjuntos train/test; sin asociacion con las etiquetas (Spearman p>0,49; chi2 p>0,20) |
| `text` | 6-306 caracteres, mediana 91; 100 % ASCII en train (sin tildes ni ene); URLs `t.co`, menciones `@usuario` y hashtags presentes |
| Nulos | 0 en todos los archivos |

Distribucion de etiquetas y combinaciones (train):

| Etiqueta | Positivos | Prevalencia | Desbalanceo (neg:pos) |
|---|---:|---:|---:|
| HS | 756 | 43,67 % | 1,29 |
| TR | 519 | 29,98 % | 2,34 |
| AG | 662 | 38,24 % | 1,62 |

| Combinacion (HS, TR, AG) | n | Comentario |
|---|---:|---|
| (0,0,0) | 949 | 54,8 % del corpus |
| (1,1,1) | 491 | 28,4 %; regimen "todo toxico" |
| (1,0,1) | 145 | odio agresivo no dirigido |
| (1,0,0) | 92 | odio no agresivo |
| (1,1,0) | 28 | dirigido sin agresividad |
| (0,0,1) | 26 | agresividad sin odio |
| (0,1,*) | 0 | **no existe**: TR=1 implica HS=1 |

Estructuras de duplicacion y agrupamiento detectadas en la auditoria:

- **4 grupos de near-duplicados** (similitud >= 0,85) entre train y test, con 7 filas de train y **10 filas de test** afectadas (`data/processed/grupos_near_duplicados.csv`). El recalculo independiente de la auditoria (TF-IDF char_wb 4-4 + coseno sobre 2.885 filas) encontro 22 pares >= 0,85, **0 pares en grupos distintos** a los guardados y 4 pares train-test; los grupos guardados son conservadores (sobre-agrupan, nunca infra-agrupan).
- **0 duplicados exactos train-test**; 5 duplicados exactos dentro de test, que incluyen un grupo de 6 textos identicos (`#?nombre?`, ids 22557/20477/24998/20081/22614/24635).
- **17 pseudo-autores** (primera mencion `@usuario`) con >= 3 tweets: 77 tweets (4,4 % de train). En el control ampliado (D-015) se identificaron 740 pseudo-autores distintos, de los cuales 39 tienen mas de un tweet.

El analisis de fuga por `id` y por duplicados no encontro mecanismos de fuga explotables mas alla del riesgo acotado de los near-duplicados y los pseudo-autores, que se abordo en la validacion (seccion 17).

## 5. EDA

El EDA profundo (`reports/eda_resumen.md`; 43 artefactos en `reports/eda/`, 26 figuras) establecio las siguientes senales, todas con tamano de efecto y correccion de Benjamini-Hochberg por familia:

**Longitud y riqueza lexica.** Los textos con TR=1 son notablemente mas cortos (delta de Cliff -0,369) que los TR=0; HS (-0,141) y AG (-0,163) muestran el mismo sentido con efecto pequeno. La deduplicacion por near-duplicados no cambio las magnitudes (TR: -0,372). La entropia lexica por clase apenas varia (HS=1: 9,062 bits vs HS=0: 9,261; TR=1: 8,752 vs TR=0: 9,321): el vocabulario toxico no es mas repetitivo en terminos globales, pero el lexico concreto si cambia. El TTR global no se uso como feature.

**Estructura de etiquetas.** Correlacion phi: HS-TR 0,743; HS-AG 0,831; TR-AG 0,759. La asociacion HS-AG es la mas fuerte entre etiquetas distintas y explica por que el par HS-AG concentra el mayor numero de errores conjuntos (seccion 20). La estructura de dos regimenes (nada toxico / todo toxico) hace que el espacio de etiquetas no sea ruido independiente.

**Fenomenos de superficie.** 47,5 % de los textos contiene al menos una mencion, 9,3 % un hashtag y 31,2 % una URL. **0 emojis Unicode** (la normalizacion ASCII de los organizadores los elimino; la via de features de emoji quedo cerrada por construccion), 54 emoticones ASCII y 19 textos con risas en patron estricto (120 con patron laxo). El corpus contiene elongaciones (`siiii`), puntuacion repetida (`!!!`) y palabras duplicadas adyacentes.

**Contraste con near-duplicados.** Al repetir todos los contrastes conservando una sola fila por grupo, el orden y la magnitud de los efectos se mantuvieron (por ejemplo, V de Cramer de `tiene_insulto` con TR: 0,339 -> 0,338). La deduplicacion elimina solo 4 filas de train: las senales no provienen de los duplicados.

**Estado inicial de las hipotesis** tras el EDA: H1 con apoyo indirecto (evasion ortografica y texto sin diacriticos), H2 parcial (lexico agresivo + segunda persona asociados a AG/HS), H3 con reservas (targeting medible, pero el riesgo es que la senal sea "es una respuesta"), H6 parcial (rasgos estilometricos significativos pero pequenos), H8 estructural (0 violaciones de la jerarquia) y H9 indirecta (prevalencias distintas de 0,5). H4, H5, H7 y H10 quedaron sin evaluar en ese momento.

## 6. Analisis linguistico

El analisis linguistico se concentro en la caracterizacion de TR (D-005), cuya definicion oficial ("dirigido contra una persona o grupo") difiere del esquema HatEval (TR como rango del objetivo: individuo vs grupo). Dentro de HS=1 (n=756: 519 TR=1 y 237 TR=0), las diferencias son grandes y sistematicas:

| Rasgo (dentro de HS=1) | TR=1 | TR=0 | V de Cramer | q |
|---|---:|---:|---:|---:|
| Mencion inicial presente | 64,9 % | 9,7 % | 0,512 | 0,00e+00 |
| Referencia colectiva presente | 10,0 % | 57,0 % | 0,504 | 0,00e+00 |
| Lexico colectivo inducido |  -  |  -  | 0,458 | 0,00e+00 |
| Segunda persona |  -  |  -  | 0,263 | 0,00e+00 |
| Insulto dirigido |  -  |  -  | 0,300 | 0,00e+00 |
| URL presente |  -  |  -  | 0,345 | 0,00e+00 |
| Imperativo |  -  |  -  | 0,030 |  -  |

Las dos lecturas de TR quedaron sobre la mesa: a favor de "rango del objetivo" estan la concentracion de respuestas directas con mencion (64,9 % vs 9,7 %) y la referencia colectiva explicita casi siempre en TR=0 (57,0 %); a favor de la definicion literal estan los 13 casos TR=0 con referencia colectiva y mencion inicial (podrian ser "dirigidos a un usuario sobre un grupo") y que solo el 68,7 % de los textos con HS=1 tiene TR=1, cuando la lectura literal "dirigido contra persona o grupo" haria esperar valores cercanos al 100 % salvo por la existencia de un regimen de odio no dirigido (noticia, cita, comentario general). La revision humana prevista en D-005 (40 ejemplos TR=1 + 40 TR=0 con rasgos anotados en `reports/eda/tr_ejemplos_para_revision.csv`) **no se completo**; el modelado uso la etiqueta tal como aparece en los datos y la ambiguedad se declara como limitacion. El error analysis refuerza la lectura del objetivo individual: varios falsos negativos de TR son textos que describen a una persona concreta ("el agresor ... inmigrante ilegal de origen Nigeriano", "este antisemita dando clase").

## 7. Textometria

La textometria midio los fenomenos de superficie por documento y los contrasto contra cada etiqueta (chi2 con V de Cramer corregida por sesgo, Fisher exacto cuando alguna frecuencia esperada < 5, diferencia de proporciones con IC95 de Newcombe y correccion BH por familia). Resumen de los fenomenos mas asociados (conjunto completo, q<0,05):

| Etiqueta | Fenomeno | V de Cramer | Sentido |
|---|---|---:|---|
| HS | `tiene_insulto` | 0,355 | mas frecuente con HS=1 |
| HS | `tiene_insulto_dirigido` | 0,173 | mas frecuente con HS=1 |
| HS | `tiene_mencion_inicial` | 0,126 | mas frecuente con HS=1 |
| HS | `tiene_hashtag` | 0,107 | mas frecuente con HS=0 |
| TR | `tiene_insulto` | 0,339 | mas frecuente con TR=1 |
| TR | `tiene_mencion_inicial` | 0,325 | mas frecuente con TR=1 |
| TR | `tiene_insulto_dirigido` | 0,266 | mas frecuente con TR=1 |
| TR | `tiene_colectivo_lexico` | 0,252 | mas frecuente con TR=0 |
| AG | `tiene_insulto` | 0,355 | mas frecuente con AG=1 |
| AG | `tiene_insulto_dirigido` | 0,195 | mas frecuente con AG=1 |
| AG | `tiene_hashtag` | 0,127 | mas frecuente con AG=0 |

La informacion mutua ajustada confirmo el mismo orden. En variables continuas por combinacion HSxTRxAG, las mas discriminantes fueron `n_insultos` (?2=0,241), `n_patron_colectivo` (?2=0,125) y `n_palabras` (?2=0,118). La textometria tambien documento los patrones de repeticion (elongaciones, puntuacion repetida) que sostienen la hipotesis de evasion ortografica H1 y que fueron el motivo de incluir n-gramas de caracter. Ninguno de estos rasgos se incorporo como bloque explicito al modelo final: entraron implicitamente via n-gramas y se usaron como variables de diagnostico en el analisis de error.

## 8. Analisis lexico

El lexico de agresividad y el inventario de referencias colectivas se indujeron **solo de train** con criterios deterministas y documentados (log-odds de lemas, frecuencia minima, presencia en varios documentos). El lexico de insultos quedo en **150 lemas**, de los cuales 19 forman el nucleo mas frecuente. El inventario colectivo incluye terminos como `inmigrante`, `ilegal`, `refugiado`.

Hallazgo metodologico central (revision `ml-reviewer` de la Fase C): estos lexicos inducidos con las etiquetas de train estan **sesgados in-sample**; toda medicion de su aporte debia inducirlos **dentro de cada pliegue** y reportar el efecto out-of-fold. Las rebanadas definidas con el lexico inducido in-sample (F1 0,368 en `tiene_insulto_lexico=0`) se excluyeron del ranking de debilidad por ser artefactos de construccion.

El efecto de los lexicos inducidos in-fold se midio en la ablacion de features densas (seccion 19): funcionan como senal (el lexico AG inducido captura 150 lemas; el colectivo captura inmigrante/ilegal/refugiado), pero **concatenarlos al TF-IDF degrado el F1** (-0,009 a -0,032). El detalle de los lexicones y su contrato de induccion quedo en `reports/features_densas_resultados.json`.

## 9. N-gramas

La comparacion de representaciones (Fase 4-5, 25 configuraciones, mismo split agrupado y mismo clasificador LR balanceada C=1.0 en las partes A y B) es el resultado experimental mas limpio del proyecto:

| ID | Representacion | Dimensiones | F1 OOF (0,5) | ? vs B1 | Pliegues |
|---|---|---:|---:|---:|---:|
| B1 / PRE-00 | TF-IDF word (1,2), min_df=2, sublinear | 4.024 | 0,7587 |  -  |  -  |
| REP-03 | TF-IDF word (1,1) | 1.953 | 0,7417 | -0,0170 | 0/5 |
| REP-01 | TF-IDF char_wb (3,5) | 23.407 | 0,7567 | -0,0021 | 2/5 |
| REP-02 | TF-IDF char_wb (2,5) | 25.036 | 0,7511 | -0,0076 | 2/5 |
| REP-04 | TF-IDF word (1,3) | 4.795 | 0,7562 | -0,0025 | 2/5 |
| **REP-05** | **union word (1,2) + char_wb (3,5)** | **27.431** | **0,7865** | **+0,0278** | **5/5** |
| REP-06 | union word (1,2) + char_wb (3,6) | 34.991 | 0,7855 | +0,0268 | 5/5 |

Lectura: **los n-gramas de caracter por si solos no superan a los de palabra** (H1 no se cumple en su forma fuerte), pero **la union si aporta de forma masiva** (H4 apoyada): +0,0278 con mejora en 5/5 pliegues. El caracter anade robustez ante la evasion ortografica, las elongaciones y la ausencia de diacriticos, mientras que la palabra conserva el lexico tematico. La configuracion ganadora definitiva (MOD-03b_BEST) sube C a 2.0 y la union pasa a 0,7927 (+0,0339 vs B1, 5/5 pliegues).

## 10. POS y sintaxis

spaCy `es_core_news_md` se uso exclusivamente como **anotador linguistico** (POS, lemas, morfologia, dependencias), con `exclude=["ner"]` y prohibicion explicita de `doc.vector`, `token.vector`, `doc.tensor` y `similarity()` en cualquier feature (D-003), cumpliendo la restriccion de no usar embeddings preentrenados. Limitaciones observadas del anotador en este corpus: spaCy asigna `Mood=Imp` a solo 63 tokens (detector de imperativo poco sensible) y no emite la etiqueta de dependencia `vocative`; por eso se combinaron tres detectores heuristicos y se reporto cada componente por separado. El NER quedo desactivado, de modo que la deteccion de nombres de persona usa una heuristica de POS, mayusculas y contexto de segunda persona.

Usos efectivos: (i) rasgos de segunda persona, imperativo, tercera persona plural y nombre propio como variables de contraste en el EDA y en el analisis de error; (ii) segmentacion de rasgos por pliegue en los bloques densos H11/H12/H14 (seccion 19). El bloque POS nunca se evaluo como representacion independiente (el experimento E05 sobre sentimiento/emocion y el bloque estilometrico E06 no se ejecutaron), por lo que la literatura que sugiere su aporte en espanol (Garibo i Orts, 2019; Garcia-Diaz et al., 2021, DOI 10.1016/j.future.2020.08.032) queda como hipotesis no probada en este proyecto.

## 11. Estilometria

Los rasgos estilometricos (longitud, puntuacion, mayusculas sostenidas, menciones, URLs, elongaciones, risas, emoticones) mostraron asociaciones **estadisticamente solidas pero de efecto pequeno** con las tres etiquetas (V de Cramer entre 0,097 y 0,141 en los rasgos de estilo; la longitud por si sola con delta de Cliff entre -0,141 y -0,369). La evidencia de la literatura sobre estilometria en textos cortos (Stamatatos, 2009, DOI 10.1002/asi.21001; Stamatatos, 2017, DOI 10.18653/v1/E17-1107) motivo la hipotesis H6, apoyada de forma parcial en el EDA.

En el modelo, los rasgos estilometricos entraron de forma implicita: los signos de puntuacion repetidos y las mayusculas se capturan parcialmente con `char_wb`, y las URLs/menciones quedan en el vocabulario de palabra. La experimentacion explicita quedo cerrada en la practica por dos resultados negativos: la variante que preserva mayusculas (PRE-01) rinde -0,0229 y la eliminacion de puntuacion (PRE-07) es neutra (0,0000), lo que indica que la informacion de estilo util ya esta representada. El bloque estilometrico denso independiente (E06) no se ejecuto y se declara como trabajo pendiente.

## 12. Lexicones

**No se uso ningun lexicon externo.** La verificacion de la auditoria confirmo que `data/external/lexicons/` esta vacio, que no hubo descargas de recursos externos y que `reports/lexicon_catalog.csv` no existe, lo cual es consistente con la decision D-007 (el catalogo solo era exigible "si se usan"). HurtLex (Bassignana et al., 2018, DOI 10.4000/books.aaccademia.3085) y los recursos afectivos tipo ML-SentiCon fueron contemplados como candidatos opcionales y finalmente **no se utilizaron ni se evaluaron**: los experimentos E02b (lexicon externo) y E04 (sentimiento/emocion) nunca se ejecutaron. La consecuencia honesta es que la hipotesis H5 (sentimiento/emocion aporta informacion complementaria) queda **sin probar**, pese a la evidencia favorable en HatEval espanol de Nina-Alcocer (2019) (`TFIDF+EMOTIONS`: EMR 0,711 vs 0,629 del baseline). Todo el lexico empleado en el proyecto es, por tanto, autocontenido: inducido de train, con contrato de induccion in-fold y trazabilidad en `reports/features_densas_resultados.json`.

## 13. Ingenieria de features

La representacion final es deliberadamente simple y fue el resultado de una ablacion sistematica:

```
texto crudo en minusculas
  -> FeatureUnion
      ?? TfidfVectorizer(analyzer='word',    ngram_range=(1,2), min_df=2,
      ?                  max_df=1.0, sublinear_tf=True)   -> 4.024 features
      ?? TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2,
                         max_df=1.0, sublinear_tf=True)   -> 23.407 features
  -> 27.431 features
  -> LogisticRegression(class_weight='balanced', C=2.0, solver='liblinear')  x3
      (una por etiqueta HS, TR, AG)
  -> umbrales por etiqueta: HS 0.44 / TR 0.51 / AG 0.46
```

Decisiones de ingenieria y su evidencia:

- **Sin eliminacion de stopwords**: la variante PRE-02 rinde -0,0025 y en la literatura las palabras funcionales son senal (MineriaUNAM usa n-gramas de stopwords 2-4).
- **Sin lematizacion**: PRE-09 rinde -0,0162; el corpus sin diacriticos degrada el lematizador y los tweets pierden senal al normalizarse (H7 apoyada).
- **Sin reconstruccion de acentos**: prohibido introducir informacion externa (D-010); el corpus es ASCII por construccion.
- **`min_df=2`**: descarta n-gramas unicos manteniendo cobertura; las sondas con `min_df=5` (10.515 features) no mejoraron la transferencia (seccion 19).
- **`sublinear_tf`**: amortigua el efecto de las repeticiones; presente en todas las variantes adoptadas.
- **Clasificadores independientes por etiqueta** con `class_weight='balanced'`: el desbalanceo moderado (maximo 2,34:1) no exigio remuestreo (H9/EDA); se descarto SMOTE/ADASYN.

Los bloques densos construidos para H11 (registro bromista, 22 features), H12 (objetivo individual vs colectivo, 15 features) y H14 (deshumanizacion inducida, 9 features) se estandarizaron con `StandardScaler` por pliegue y se concatenaron a la union en 46 columnas; su resultado fue negativo y no entraron al modelo final (seccion 19). El test nunca participo en ningun ajuste: la auditoria verifico que el vocabulario del artefacto final coincide exactamente con un ajuste desde cero solo con train (27.431 = 4.024 + 23.407).

## 14. Baselines

Se establecieron tres referencias y una sonda:

1. **Sonda E00 (identificacion de la metrica)**: se envio la plantilla oficial todo-1 (`submissions/toxic_submission_v00_plantilla.csv`). LB observado **0,51494**. Los valores esperados con la distribucion de train eran 0,541 (media column-wise de la F1 binaria), 0,543 (micro), 0,270 (macro sobre 6 pares) y 0,284 (samples/exact match). La observacion descarta macro-6 y samples; la confirmacion final llego con el baseline: local 0,75778 (media column-wise) vs LB 0,75677 (diferencia -0,0010), mientras el macro-6 local era 0,8097. **Metrica oficial = F1 macro sobre las 3 columnas.**
2. **B1 (baseline clasico)**: TF-IDF word (1,2) min_df=2 sublinear + LR balanceada C=1.0, 4.024 features. OOF 0,7587 (sin post-proceso) / 0,7578 (con post-proceso jerarquico); por etiqueta HS 0,743 / TR 0,803 / AG 0,731; exact match 0,701; 1,14 s de entrenamiento. LB 0,75677.
3. **B5-DENSE (baseline denso)**: LR C=1.0 sobre los 46 rasgos densos H11+H12+H14 estandarizados, sin TF-IDF. OOF 0,6607. Sirve de control: los rasgos densos *per se* contienen senal, pero muy inferior a la representacion de n-gramas.
4. **Sonda de la plantilla como suelo**: 0,51494 (todo positivo) delimita el comportamiento de un clasificador trivial en la metrica oficial.

El baseline B1 ya revelo el fenomeno que dominaria el proyecto: el modelo **subpredice positivos** con umbral 0,5 (HS predicho 0,373 vs prevalencia 0,437; AG 0,310 vs 0,382), lo que abrio la linea de umbrales por etiqueta.

## 15. Modelos clasicos

El barrido de la Parte C comparo clasificadores lineales sobre la representacion ganadora (mismo split, mismo protocolo):

| ID | Modelo | F1 OOF | ? vs B1 | Pliegues | Decision |
|---|---|---:|---:|---:|---|
| MOD-02_BEST | ComplementNB(alpha=1.0) | 0,6888 | -0,0699 | 0/5 | Descartado |
| MOD-03a_BEST | LR C=0.5, liblinear, balanced | 0,7723 | +0,0135 | 5/5 | Adoptado como referencia |
| MOD-01_BEST | LinearSVC C=1.0, balanced | 0,7849 | +0,0262 | 5/5 | Adoptado como base de ensemble |
| REP-05 | LR C=1.0 sobre la union | 0,7865 | +0,0278 | 5/5 | Adoptado |
| **MOD-03b_BEST** | **LR C=2.0, liblinear, balanced** | **0,7927** | **+0,0339** | **5/5** | **Modelo final** |
| MOD-04 | MOD-03b + post-proceso jerarquico | 0,7870 | +0,0283 | 5/5 | Descartado (-0,0056 vs su origen) |

El mensaje es consistente con Joachims (1998, DOI 10.1007/BFb0026683): en clasificacion de texto con representaciones dispersas de alta dimension, un modelo lineal regularizado es competitivo; la familia Naive Bayes queda muy por detras al no modelar las interacciones de evidencia negativa. La regularizacion importa: C=2.0 supera a C=1.0 y a C=0.5 en OOF, pero el analisis posterior (D-013) mostraria que **mas capacidad local no implico mejor transferencia al test**  -  la mejora de C=2.0 sobre C=1.0 no se reprodujo en el leaderboard (v03 0,78306 vs v06 0,77227, aunque con umbrales distintos entre ambas).

## 16. Deep learning desde cero

La Fase 11 fue un **requisito metodologico**, no una palanca de rendimiento (D-014). Se entrenaron tres arquitecturas desde cero, con vocabulario y preprocesamiento ajustados unicamente con el pliegue de entrenamiento, cabezal multi-tarea de 3 logits, `BCEWithLogitsLoss` con `pos_weight` por etiqueta, AdamW (lr 1e-3, wd 1e-4), batch 32, maximo 30 epocas y early stopping estricto sobre la F1 de validacion (paciencia 5, mejores pesos restaurados):

| ID | Arquitectura | Parametros | F1 OOF | F1 por etiqueta (HS/TR/AG) | Gap train-val | Decision |
|---|---|---:|---:|---:|---:|---|
| DL-01 | MLP sobre SVD(300) del TF-IDF union, capas 256-128 | 110.339 | **0,7949 +/- 0,0262** | 0,7838 / 0,8217 / 0,7790 | +0,1854 | Supera a MOD-03b en OOF (+0,0022); no adoptado (D-012/D-014) |
| DL-01-SEEDS | Idem, semillas 42/43/44 | 110.339 | 0,7947 +/- 0,0011 |  -  |  -  | Estable entre semillas |
| DL-02 | CharCNN (embeddings de caracter dim 48, kernels 3 y 5) | 87.059 | 0,7410 +/- 0,0097 | 0,7093 / 0,7763 / 0,7375 | +0,1807 | Descartado (peor que B1) |
| DL-03 | BiLSTM de palabra (embeddings dim 64, vocab 1.685, seq 40) | 233.411 | 0,7364 +/- 0,0199 | 0,7237 / 0,7862 / 0,6992 | +0,1896 | Descartado (peor que B1) |

Lectura honesta: la mejor red (DL-01) alcanza 0,7949 de F1 OOF frente a 0,7927 del modelo clasico, una diferencia de +0,0022 que **la regla del proyecto no permite adoptar** (umbral +0,005) y que ademas carece de sonda de leaderboard: por D-012/D-014 no se gasto un envio. El MLP opera sobre una proyeccion SVD del mismo TF-IDF, de modo que su ventaja no proviene de una representacion nueva sino de una frontera de decision no lineal sobre la misma informacion. Los modelos secuenciales puros (CharCNN, BiLSTM) quedan por debajo incluso del baseline B1: con 1.731 ejemplos de entrenamiento, vocabularios diminutos y prohibicion de embeddings preentrenados, el costo inductivo de las redes no se paga. Este resultado coincide con la advertencia de la literatura sobre el tamano de datos en deteccion de odio (Wiegand et al., 2019) y con la evidencia de que los modelos lineales sobre n-gramas siguen siendo muy competitivos en corpus pequenos.

Nota tecnica: AMP (autocast fp16) **no se uso** porque, medido en la GPU disponible (AMD RX 9070 XT, ROCm), el costo de `autocast` superaba la ganancia (BiLSTM: 3,42 s vs 7,48 s en 3 epocas). El protocolo de acceso a GPU compartida con el escritorio (D-008) se aplico antes de cada job y quedo en `experiments/gpu_preflight.log`; el test nunca se cargo en el script de DL (0 ocurrencias de `read_test`, verificado por la auditoria).

## 17. Validacion

El protocolo de validacion evoluciono con la evidencia:

**Protocolo primario.** `StratifiedGroupKFold(k=5, shuffle=True, random_state=42)` con grupos = componentes conexas de similitud >= 0,85 (near-duplicados) y estratificacion por la combinacion HSxTRxAG. La reconstruccion independiente de la auditoria confirmo **0 grupos compartidos entre train y validacion** en los 5 pliegues y **0 textos de validacion con copia exacta en train** (normalizado). El vectorizador, el escalador y los lexicos inducidos se ajustan dentro de cada pliegue (`fit_transform` solo sobre el indice de entrenamiento); el test solo se usa con `transform`/`predict`.

**Control por pseudo-autor (D-015).** La auditoria final detecto que el control previsto en D-004 nunca se habia ejecutado: con el split solo por near-duplicados, **17/17 pseudo-autores con >=3 tweets cruzaban pliegues** (por ejemplo, `@youtube` aparecia en los 5). Al fusionar los grupos (near-duplicado + pseudo-autor) y reevaluar el modelo final con el mismo protocolo, la F1 OOF baja de 0,7927 a **0,7838** (-0,0088) y el **gap local-LB pasa de -0,0128 a -0,0039** comparado con el LB real de esa configuracion (v02 = 0,77993). Desviacion entre pliegues del control fusionado: 0,0187, frente a 0,0088 del split por near-duplicados. Conclusion: el split solo por near-duplicados era optimista porque un mismo usuario produce tweets con etiquetas correlacionadas; el protocolo recomendado para trabajo futuro es el agrupamiento fusionado. La seleccion del modelo final no se reabrio (D-014), pero el hallazgo se documenta como la explicacion principal del gap.

**Evaluacion de umbrales y ensembles.** Los umbrales se optimizaron solo sobre probabilidades OOF (rejilla 0,20-0,80 paso 0,01, por etiqueta) y la ganancia optimista (+0,0063) se contrasto con una verificacion anidada independiente: **+0,0014 de media, 2/5 pliegues positivos**. Los ensembles se evaluaron con protocolo leave-one-fold-out (pesos/meta-modelo ajustados en 4 pliegues y evaluados en el restante), con la limitacion declarada de que las probabilidades base son OOF compartidas.

**Auditoria de fuga.** Aprobada con observaciones: sin uso de corpus/etiquetas externas, sin embeddings preentrenados, sin seleccion de features/umbrales sobre test, sin procesos de envio automatico. Dos hallazgos relevantes: (i) la brecha del pseudo-autor (arriba); (ii) los grupos de near-duplicados guardados son conservadores. El vocabulario del artefacto final coincide exactamente con un ajuste solo-train; las submissions v01, v02 y v03 se reprodujeron **bit a bit** (sha256 identicos) en una copia aislada.

**Reproducibilidad.** Semillas `random_state=42` en todos los scripts clasicos; DL con semillas 42/43/44; resultados deterministas (dos ejecuciones de `04_refit_submissions_v02_v03.py` en copia aislada produjeron la misma metrica y las mismas submissions). 6/6 modelos joblib, 5/5 JSON de metricas y 3/3 NPZ OOF cargan con formas correctas.

## 18. Experimentos

Se registraron **51 experimentos** en `experiments.csv` (52 lineas con cabecera) con pregunta, hipotesis, preprocesamiento, representacion, modelo, hiperparametros, protocolo de validacion, metrica, interpretacion y decision. La siguiente tabla sintetiza los hitos por fase:

| Fase | ID | Experimento | F1 OOF | ? vs referencia | Decision |
|---|---|---|---|---:|---|
| Metrica | E00 | Sonda de plantilla todo-1 |  -  | LB 0,51494 | Metrica identificada (D-009) |
| Baseline | B1 | TF-IDF word + LR | 0,7587 |  -  | Referencia adoptada |
| Preproc. | PRE-03 | URLs -> token `url` | 0,7623 | +0,0035 | Descartada (< 0,005) |
| Preproc. | PRE-09 | Lematizacion spaCy | 0,7425 | -0,0162 | Descartada |
| Repres. | REP-05 | Union word+char | 0,7865 | +0,0278 | Adoptada |
| Modelos | MOD-03b | LR C=2.0 sobre la union | **0,7927** | +0,0339 | **Modelo final** |
| Post-proc. | MOD-04 | HS=0 ? TR=0, AG=0 | 0,7870 | -0,0056 | Descartada (H8) |
| Umbrales | TH-01 | Umbrales OOF 0,44/0,51/0,46 | 0,7989 | +0,0063 (optimista) | Adoptada como apuesta (D-012) |
| Ensembles | ENS-01..04 | Votacion, pesos, stacking, mejor par | 0,7906-0,7985 | max. +0,0058 (max. de 12) | Descartados (< +0,005 anidado; v04 LB 0,77510) |
| Densas | HIB-H12 | Union + objetivo individual/colectivo | 0,7836 | -0,0091 | Descartada (H12) |
| Densas | HIB-H11 | Union + registro bromista | 0,7713 | -0,0213 | Descartada (H11) |
| Densas | HIB-H14 | Union + deshumanizacion | 0,7630 | -0,0297 | Descartada (H14) |
| Densas | HIB-UNION | Union + 46 densas | 0,7610 | -0,0316 | Descartada |
| Densas | B5-DENSE | Solo 46 densas | 0,6607 | -0,1319 | Descartada |
| Optuna | OPT-01 | Mejor trial (50 trials, 28 completos) | 0,8012 (0,8039 con umbrales) | +0,0086 | Adoptado localmente; v05 LB 0,77602 -> descartado |
| Robustez | R1-C1 | C=1.0 | 0,7929 | +0,0002 | v06 LB 0,77227 -> descartada |
| Robustez | R2-MINDF5 | min_df=5, C=1.0 | 0,7993 | +0,0064 | v07 LB 0,77034 -> descartada |
| DL | DL-01 | MLP sobre SVD(300) | 0,7949 | +0,0022 | Requisito metodologico; no adoptado |
| DL | DL-02/03 | CharCNN / BiLSTM | 0,7410 / 0,7364 | -0,052 / -0,056 | Descartadas |
| Validacion | PA-01 | Control por pseudo-autor | 0,7838 | -0,0088 | Protocolo recomendado (D-015) |
| v2 | S08 | Sonda v08: meta de stacking v2 | 0,7874 (dev) | LB 0,78163 | No adoptada (-0,0014 vs v03) |
| v2 | CIERRE | Investigacion v2 (24 lineas, P0-P7) |  -  |  -  | Sin mejora transferible; modelo final V03 |

Patron general: de las mas de 50 configuraciones evaluadas, **una sola familia** (union word+char con LR lineal y umbrales) se adopto como modelo final; todas las vias alternativas  - preprocesamiento linguistico, features densas, ensembles, optimizacion de hiperparametros, control de capacidad y redes desde cero -  produjeron mejoras locales nulas, negativas o no transferibles. Este patron no es un fracaso del proceso sino su resultado principal: la representacion de n-gramas con un clasificador lineal regularizado es muy dificil de superar en este regimen de datos y restricciones.

## 19. Ablacion

La ablacion permitio separar que componente aporta y que no. Componentes evaluados con el mismo protocolo agrupado:

| Componente | Variante | Efecto medido | Veredicto |
|---|---|---|---|
| Preprocesamiento (10 variantes) | Mejor: URLs -> `url` (PRE-03) | +0,0035 (< 0,005) | Descartado |
| Preprocesamiento | Lematizacion | -0,0162 | Descartado (H7 apoyada) |
| Preprocesamiento | Preservar mayusculas | -0,0229 | Descartado |
| Preprocesamiento | Quitar stopwords | -0,0025 | Descartado |
| Preprocesamiento | Eliminar puntuacion | 0,0000 | Neutro |
| Representacion | char_wb sola | -0,0021 vs word sola | H1 no confirmada en forma fuerte |
| Representacion | Union word+char | +0,0278 (5/5 pliegues) | **Adoptado** (H4) |
| Capacidad | C=2.0 -> C=1.0 | OOF -0,0062; LB -0,0108 | Descartado (D-013) |
| Capacidad | min_df 2 -> 5 | OOF +0,0004 con umbrales; LB -0,0127 | Descartado (D-013) |
| Post-proceso | HS=0 ? TR=0, AG=0 | -0,0056 / -0,0004 (solo TR) | Descartado (H8) |
| Umbrales | Por etiqueta vs 0,5 | OOF +0,0063; LB +0,0031 pareado; anidado +0,0014 | Adoptado (D-012, H9) |
| Densas | H11 / H12 / H14 / union | -0,0213 / -0,0091 / -0,0297 / -0,0316 | Descartado (H11, H12, H14) |
| Ensemble | 4 estrategias | max. +0,0023 anidado; v04 LB -0,0080 vs v03 | Descartado (H10) |
| DL | 3 arquitecturas | max. +0,0022 OOF sin sonda | No adoptado |
| Validacion | Grupos fusionados con pseudo-autor | -0,0088 (calibracion, no rendimiento) | Adoptado como protocolo |

Un control especialmente informativo: al poner las 46 columnas densas a cero dentro del pipeline hibrido, la metrica reproduce exactamente 0,792659 (la de MOD-03b). Es decir, **la perdida de los hibridos es atribuible a las features densas, no a la concatenacion**, lo que descarta errores de implementacion como causa del resultado negativo. De forma analoga, el control REF-MOD03b de las sondas de robustez reprodujo 0,792659 exacto y probabilidades identicas a `features/oof_mod03b.npz` (diferencia maxima 0.0e+00): el arnes experimental es fiel.

## 20. Analisis de error

El analisis de error (`reports/error_analysis.md`) uso las probabilidades OOF del modelo final con los umbrales adoptados (HS 0,44 / TR 0,51 / AG 0,46). Errores totales: **425 falsos positivos y 375 falsos negativos** sobre 3x1.731 decisiones; 251 ejemplos fallan una etiqueta, 123 fallan dos y **101 (5,8 %) fallan las tres**. Con umbral 0,5 el balance era 349 FP / 445 FN: los umbrales trasladan error de FN a FP.

Matrices de confusion (escenario de umbrales adoptados):

| Etiqueta | Umbral | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HS | 0,44 | 614 | 185 | 142 | 790 | 0,769 | 0,812 | **0,790** |
| TR | 0,51 | 431 | 89 | 88 | 1123 | 0,829 | 0,830 | **0,830** |
| AG | 0,46 | 517 | 151 | 145 | 918 | 0,774 | 0,781 | **0,777** |

TR es la etiqueta mejor resuelta; HS la de mas errores absolutos (327); AG la de menor F1. La regla estructural `HS=0 ? TR=0` vale -0,0004 y su version completa -0,0049: no es una palanca (H8 cerrada).

**Donde falla el modelo.** Las 5 rebanadas mas debiles (n >= 40, excluyendo artefactos in-sample del lexico):

| # | Rebanada | n | F1 HS | F1 TR | F1 AG | F1 medio | ? vs global |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `tiene_hashtag=1` | 161 | 0,622 | 0,545 | 0,590 | **0,586** | -0,213 |
| 2 | `tiene_pregunta=1` | 158 | 0,682 | 0,653 | 0,563 | 0,633 | -0,166 |
| 3 | `tiene_referencia_colectiva=1` | 418 | 0,697 | 0,644 | 0,576 | 0,639 | -0,160 |
| 4 | cuartil de longitud Q4 | 432 | 0,696 | 0,629 | 0,611 | 0,645 | -0,154 |
| 5 | `tiene_mencion=0` | 909 | 0,703 | 0,707 | 0,681 | 0,697 | -0,102 |

Regiones del corpus: `responde_mencion_inicial` (n=701) F1 0,876; `regimen_noticia` (URL o hashtag, n=599) F1 0,702; `no_responde_con_colectiva` (n=318) F1 0,597. La agresividad que se escapa es contra grupos: el **45,5 % de los FN de AG tiene referencia colectiva frente al 13,9 % de los TP**. El error de TR vive fuera de las respuestas: F1_TR 0,875 con mencion inicial vs 0,597 sin mencion con colectiva.

**Patrones cualitativos** (revision de 90 muestras en `reports/eda/error_samples_para_revision.csv`):

1. **Insulto interpersonal en registro de broma o afiliacion** (FP dominante de HS, TR y AG): de 526 textos con insulto, mencion inicial y sin referencia colectiva, **213 (40,49 %) estan etiquetados (0,0,0)**. El modelo dispara con probabilidad 0,87-0,97 textos como `@CamilaArvayo Cllate todo te queda bien zorra te amo`, porque el vocativo, los posesivos afectivos ("mi novia", "mi perra"), la risa (`JAJAJA`) y las marcas de juego desactivan la lectura toxica en la anotacion, pero un modelo de bolsas de n-gramas suma evidencia de insulto y no representa conjunciones con marcadores de afiliacion.
2. **Uso metalinguistico: cita o reproduccion del insulto** (FP y FN): el modelo lineal reacciona al lexico citado tanto cuando la cita es inocua como cuando la cita es el vehiculo del odio.
3. **Regimen noticia o atribucion a terceros** (FN dominante): textos largos con URL que describen violencia contra un grupo; varios describen a un individuo concreto, lo que refuerza la lectura de D-005.
4. **Odio implicito sin insulto del lexico**: estereotipo ("El Arabe es un flojo"), miedo estigmatizante ("m da miedo cuando me sigue un rabe"), deshumanizacion ("una linda gallina que acabar en un KFC"). El modelo necesita el insulto explicito para activar HS/AG.
5. **Frontera de anotacion no observable en el texto**: casos de superficie casi identica con etiquetas opuestas; el 40,49 % de la region de insulto interpersonal es (0,0,0). Este limite acota el techo alcanzable con cualquier modelo de texto.

Del analisis surgieron cinco hipotesis nuevas (H11-H15) con experimento y criterio; su ejecucion posterior fue predominantemente negativa (H11 -0,0213; H12 -0,0091; H14 -0,0297), y **H13 (cita/atribucion) y H15 (umbrales por rebanada) nunca se ejecutaron**; se declaran como trabajo futuro.

## 21. Modelo final

**MOD-03b_BEST + umbrales por etiqueta** (decision D-014, cerrada el 2026-09-23):

- **Representacion**: `FeatureUnion` de TF-IDF word (1,2) min_df=2 max_df=1.0 sublinear_tf (4.024 features) + TF-IDF char_wb (3,5) min_df=2 max_df=1.0 sublinear_tf (23.407 features) = **27.431 features**.
- **Modelo**: tres `LogisticRegression(class_weight='balanced', C=2.0, solver='liblinear', random_state=42, max_iter=1000)`, una por etiqueta, sin post-proceso jerarquico.
- **Umbrales de decision** (optimizados solo sobre OOF): HS **0,44**, TR **0,51**, AG **0,46**.
- **Metricas OOF**: F1 macro columnas **0,7989** (0,7927 con umbral 0,5); F1 por etiqueta HS 0,7897 / TR 0,8296 / AG 0,7774; macro-6 0,8356; micro 0,7898; exact match 0,7314 (umbral 0,5); desviacion entre pliegues 0,0088.
- **Leaderboard**: **0,78306** (submission v03), el mejor de 8 envios; gap local-LB -0,0158 con el split original, **-0,0039** con el control por pseudo-autor (D-015).

**Por que se mantienen los umbrales pese a la verificacion anidada modesta**: el contraste pareado v02 (umbral 0,5) vs v03 (umbrales) en el mismo modelo dio **+0,0031 real en el leaderboard**, dentro del rango entre la estimacion anidada (+0,0014) y la optimista (+0,0063). Se prohibio re-optimizar umbrales contra el LB. La submission v02 se conserva como control metodologico (LB 0,77993).

**Artefactos**: `models/mod03b_lr_union.joblib` (vectorizador + 3 modelos + umbrales), `features/oof_mod03b.npz` (probabilidades OOF), `submissions/toxic_submission_v03_lr_union_thr.csv` (sha256 `7d7478f271e6edfd...`), script `experiments/04_refit_submissions_v02_v03.py`. La auditoria verifico que el artefacto reproduce v03 exactamente (0 filas distintas) y que la submission se reconstruye byte a byte desde el script en copia aislada.

**Plateau (D-014)**: los 6 modelos de la familia union evaluados en el LB caen en la banda **0,77034-0,78306**, un rango de 0,0127 = **0,99 errores estandar** (SE binomial ? 0,0129 con n=1.154). Las diferencias entre variantes son indistinguibles del ruido de muestreo; el unico contraste pareado fiable es el de umbrales. Por eso se detuvo el sondeo del leaderboard y se redirigio el esfuerzo a los entregables academicos y a completar la Fase 11 como requisito metodologico.

## 22. Submission

Se enviaron 8 submissions, todas validadas contra la plantilla oficial (columnas `id,HS,TR,AG`, 1.154 filas, ids identicos y en orden, valores en {0,1}, sin nulos ni duplicados) y **enviadas manualmente por Carlos** (0 llamadas a la API/CLI de envio en el codigo). Registro completo en `submissions/leaderboard_log.csv`:

| Envio | Modelo | Local (OOF) | Kaggle | Gap | Positivos HS/TR/AG (%) |
|---|---|---:|---:|---:|---|
| v00 plantilla | Todo 1 |  -  | 0,51494 |  -  | 100 / 100 / 100 |
| v01 baseline | TF-IDF word + LR | 0,75778 | 0,75677 | -0,0010 | 37,4 / 27,9 / 31,0 |
| v02 union u=0,5 | MOD-03b, umbral 0,5 | 0,79270 | 0,77993 | -0,0128 | 38,0 / 29,3 / 34,2 |
| **v03 union + umbrales** | **MOD-03b + 0,44/0,51/0,46** | **0,79890** | **0,78306** | -0,0158 | 43,9 / 28,9 / 36,6 |
| v04 ensemble mediana | ENS-04 (M2+M3), umbrales medianos | 0,79850 | 0,77510 | -0,0234 | 46,5 / 30,2 / 40,5 |
| v05 optuna | OPT-01 + umbrales | 0,80390 | 0,77602 | -0,0279 | 41,0 / 29,7 / 35,8 |
| v06 robustez C=1.0 | MOD-03b C=1.0 + umbrales | 0,79290 | 0,77227 | -0,0206 | 44,5 / 28,2 / 38,6 |
| v07 robustez min_df=5 | min_df=5, C=1.0 + umbrales | 0,79930 | 0,77034 | -0,0290 | 45,1 / 29,2 / 35,4 |

Hallazgos de la fase de submission:

- La sonda E00 fijo la metrica y revelo que el test tiene del orden de **7 % menos positivos** que train (factor ? 0,93): el valor 0,51494 esta por debajo del 0,541 esperado con la prevalencia de train.
- **El mejor OOF no fue el mejor LB**: OPT-01 (0,8039 local) obtuvo 0,77602; MOD-03b (0,7989) obtuvo 0,78306. El gap local-LB crecio con la capacidad del modelo (B1 -0,0010; MOD-03b -0,0128/-0,0158; ENS-04 -0,0234; OPT-01 -0,0279).
- **El gap train-OOF no predice el gap local-LB** (D-013 refutada): reducir C e incrementar min_df redujo el gap train-OOF (0,1818 -> 0,1635 -> 0,1272) pero empeoro el gap local-LB (-0,0206 y -0,0290).
- El ensemble con umbrales medianos predijo mas positivos que la prevalencia de train y cayo (0,77510); el patron observado es que sobredisparar la prevalencia del test penaliza, aunque la evidencia es insuficiente para convertirlo en regla (v05, con prevalencia cercana a la estimada, tampoco supero a v03).

**Estado final**: la submission vigente y de respaldo es **v03** (LB 0,78306). Fallback secundario: v02 (control sin umbrales, 0,77993); ultimo recurso v01 (0,75677). Las variantes v04-v07 permanecen en disco como evidencia y no deben reenviarse.

## 23. Limitaciones

1. **Tamano y calidad del corpus**: 1.731 ejemplos de entrenamiento para 27.431 features y tres tareas; 101 ejemplos (5,8 %) fallan las tres etiquetas y la frontera de anotacion entre insulto interpersonal y odio no es observable en el texto (40,49 % de la region de insulto + mencion sin colectivo esta etiquetada como (0,0,0)). Cualquier modelo de texto tiene un techo acotado por esta ambiguedad (Wiegand et al., 2019; Rottger et al., 2021, DOI 10.18653/v1/2021.acl-long.4).
2. **Corpus normalizado a ASCII**: sin tildes ni ene, con 6 bytes `0xBF` en test; parte de la senal morfologica del espanol es irrecuperable y no se debe reconstruir con recursos externos (D-010).
3. **Gap local-LB**: la cota local fue optimista entre -0,001 (B1) y -0,029 (v07). Con el protocolo de grupos fusionados (D-015) el gap se reduce a -0,0039 para el modelo final, pero ese protocolo no se uso para la seleccion original.
4. **Plateau**: las diferencias entre variantes de la familia union (0,77034-0,78306) equivalen a ~1 error estandar; el ranking del leaderboard a este nivel es en gran parte ruido de muestreo.
5. **Pseudo-autores**: 17 pseudo-autores (77 tweets, 4,4 %) cruzaban pliegues en el split original; la validacion sin ellos era optimista en ~0,0088. El protocolo recomendado (grupos fusionados) se documento, pero no se volvio a seleccionar modelo con el.
6. **TR sin resolver (D-005)**: la semantica de TR sigue ambigua entre la definicion oficial y el esquema HatEval; la revision humana prevista no se completo. El modelo usa la etiqueta tal como aparece.
7. **Hipotesis y experimentos no ejecutados**: H5 (sentimiento/emocion) con E02b/E04/E05; H13 (cita/atribucion); H15 (calibracion por rebanada); el bloque estilometrico/POS E06. Se declaran como alcance no cubierto, no como resultados negativos.
8. **DL-01 sin sonda de leaderboard**: supero a MOD-03b en OOF (+0,0022) y no se envio; la omision es una decision explicita por D-012/D-014 (regla de +0,005 y cierre de la fase competitiva), no una omision silenciosa.
9. **Near-duplicados train-test**: 10 textos de test tienen near-duplicados en train; la particion agrupada lo mitiga, pero el escenario del LB incluye esos casos y la estimacion local no es perfectamente comparable.
10. **Estimacion indirecta de la prevalencia del test**: el factor ?0,93 se infiere de la sonda E00 y de los gaps, no de las etiquetas de test (prohibido usarlas).
11. **Una sola semilla de particion**: `random_state=42` en todos los experimentos clasicos; la robustez se evaluo via variantes y el control por pseudo-autor, no via multiples particiones (los ensembles/densas si usaron esquemas anidados).
12. **30 % de la nota depende del notebook**: el entregable `notebooks/10_final_pipeline.ipynb` (31 celdas, todas las de codigo ejecutadas) existe y consolida la narrativa, pero su evaluacion es independiente de este informe.
13. **Limites evidenciados en la fase v2**: los deltas locales que respaldaron al candidato (+0,0081 en desarrollo; +0,0180 en el holdout de 248 tweets) son del orden del error estandar del test (~0,013 con n=1.154; el holdout tiene EE ~0,03), por lo que no garantizaban transferencia; el gap dev-LB de la familia union es negativo (-0,013 a -0,016) y se confirmo en todas sus variantes; y con 1.154 filas la comparacion en el test no distingue V03 del meta (diferencia -0,0014 frente a un error estandar ~0,013), de modo que la unica lectura que resiste es que ambos son indistinguibles en el test.

## 24. Conclusiones

1. **La metrica correcta se determino empiricamente**, no por supuesto: F1 macro sobre las 3 columnas (media column-wise de la F1 binaria), confirmada por la sonda E00 (0,51494) y por el baseline (local 0,75778 vs LB 0,75677). El protocolo local reproduce el LB con error ~0,001 en configuraciones de baja capacidad.
2. **La representacion de n-gramas con un clasificador lineal regularizado es el nucleo ganador** en este regimen de datos y restricciones: union word(1,2)+char_wb(3,5) con LR C=2.0 aporta +0,0339 sobre el baseline B1 con mejora en 5/5 pliegues, y ninguna de las mas de 50 configuraciones evaluadas logro superarla de forma transferible.
3. **Los umbrales por etiqueta aportaron una mejora real, pequena y pareada** (+0,0031 en el LB, mismo modelo): es el unico contraste del leaderboard que supera el ruido, coherente con la literatura de umbral optimo para F1 (Lipton et al., 2014; Nan et al., 2012).
4. **Los resultados negativos son evidencia de primera clase**: preprocesamiento linguistico, features densas, ensembles, Optuna, control de capacidad y DL desde cero no superaron la barrera pre-registrada; varios de ellos (los hibridos densos) fueron controlados experimentalmente para descartar artefactos de implementacion.
5. **Dos hallazgos metodologicos transferibles**: (i) el gap train-OOF no predice el gap local-LB (D-013 refutada) porque el optimismo de la validacion agrupada es de otra naturaleza que la memorizacion in-sample; (ii) el pseudo-autor explica la mayor parte del gap optimista del protocolo (D-015: gap -0,0128 -> -0,0039), por lo que el agrupamiento debe fusionar near-duplicados y autores.
6. **La fase competitiva alcanzo su plateau** con los datos y restricciones disponibles: 6 modelos en una banda de ~1 error estandar. La decision D-014 de detener el sondeo fue la respuesta correcta al riesgo de sobreajustar el leaderboard.
7. **Las restricciones se cumplieron integramente** y quedaron verificadas por auditoria independiente: sin transformers/LLM, sin embeddings preentrenados, sin datos externos, test nunca usado para entrenar, submissions en formato valido, resultados reproducibles bit a bit. Veredicto GO.
8. **El resultado final** (LB 0,78306, OOF 0,7989 con umbrales y 0,7838 con el protocolo corregido por pseudo-autor) es consistente con el estado del arte clasico en espanol: el mejor sistema sin transformers de HatEval espanol logro macro-F1 0,73 (Argota Vega et al., 2019) en un corpus binario de odio, y este proyecto opera en tres etiquetas simultaneas con metrica macro.

## 25. Trabajo futuro

1. **Adoptar el protocolo de grupos fusionados (near-duplicado + pseudo-autor)** como estandar de validacion en cualquier extension del trabajo; reportar siempre ambas estimaciones (D-015).
2. **Ejecutar las hipotesis pendientes** con la regla de decision pre-registrada: H13 (features de cita/atribucion para el regimen noticia), H15 (umbrales por rebanada, como experimento de cierre), y H5/E02b/E04/E05 (sentimiento/emocion y lexicones externos con catalogo y licencia, manteniendo una variante autocontenida).
3. **Atacar los dos regimenes de error por separado**: (i) registro de respuesta con interacciones explicitas insulto x afiliacion (la linea H11 necesita otra formulacion: regularizacion por bloque, modelos con conjunciones o calibracion condicionada por subgrupo); (ii) regimen noticia/atribucion con features de metalenguaje y estructura del titular.
4. **Explorar el MLP (DL-01) como fuente de diversidad real** frente al lineal: pese a que su OOF no es adoptable sin sonda, su frontera no lineal podria ensamblarse mejor con la del LR que los ensembles de modelos correlacionados, previa validacion anidada rigurosa.
5. **Calibracion probabilistica** (Platt/Isotonic) y estudio de costos asimetricos si el caso de uso final pondera de forma distinta los FP y los FN de odio.
6. **Validacion externa controlada**: repetir el pipeline en otro corpus espanol de odio con el mismo esquema, verificando que la representacion y los umbrales transfieren, sin usar etiquetas externas en el entrenamiento del modelo de la competencia.
7. **Anotacion y estudio de la frontera**: cuantificar el techo real con una muestra doblemente anotada en la region de insulto interpersonal (donde el desacuerdo entre anotadores acota el rendimiento alcanzable) y cerrar la revision de D-005 con anotadores humanos.
8. **Notebook y documentacion**: mantener el notebook como fuente reproducibilidad de punta a punta (regeneracion de la submission v03 verificada bit a bit) y ampliar la discusion didactica de los resultados negativos, que son el mayor activo formativo del proyecto.
9. **Ejecutar las lineas del backlog v2 no cubiertas** (5 de 24; las demas se evaluaron y cerraron en la fase v2): calibracion de probabilidades (Platt/isotonic en OOF anidado, E2-14), encoder de entidades (target encoding OOF de menciones/hashtags/dominios, exigiendo cobertura > 20 %, E2-13), especialistas por rebanada (E2-20), umbrales por rebanada con shrinkage (E2-16, H15) y espacios de features por etiqueta (E2-21), cada una con la regla de adopcion pre-registrada de D-016.
10. **Adoptar el protocolo de validacion corregido de la v2 en cualquier trabajo derivado**: grupos fusionados (near-duplicado UNION pseudo-autor) + holdout congelado con presupuesto explicito de evaluaciones + bootstrap por tweet, reportando siempre la incertidumbre de los deltas y evitando tratar el maximo observado como objetivo reproducible (winner's curse).

## 26. Investigacion posterior (fase v2)

Fecha de la fase: 2026-09-25. Estado: **concluida sin mejora robusta transferible al test** (condicion de terminacion B del plan v2: las fuentes razonables de senal fueron evaluadas y documentadas). Fuentes: `reports/advanced_research_v2.md` (informe de cierre), `DECISIONS.md` (D-016 a D-020), `tasks/todo.md` (gates P0-P7) y `notebooks/10_final_pipeline.ipynb` (secciones 9-11 y 14).

Contexto: tras cerrar la fase competitiva (D-014), se ejecuto una segunda ronda de investigacion de **24 lineas** para intentar superar el LB 0,78306 de V03. A diferencia de la ronda historica, toda comparacion se hizo con un protocolo de validacion corregido y una regla de adopcion pre-registrada.

### 26.1 Protocolo de validacion corregido (D-016)

- **Grupos** = near-duplicado **UNION** pseudo-autor (grupos fusionados), centralizados en `src/splits.py`.
- **Holdout congelado**: 1 pliegue de `StratifiedGroupKFold(n_splits=7, seed=42)` = **248 tweets** (los 6 combinaciones de etiquetas representadas), con sha256 verificado (`e5b62e5f...`) y presupuesto maximo de **5 evaluaciones**, de las cuales se usaron **2**. El resto del train es el **desarrollo (1.483 tweets)**.
- **Metrica primaria**: `corrected_cv` sobre el desarrollo (`historical_cv` se reporta para continuidad). Referencia v2: **V03 con umbrales = 0,7772** (M2 con umbral 0,5 = 0,7718; por etiqueta HS 0,762 / TR 0,816 / AG 0,754).
- **Regla de adopcion pre-registrada**: corrected_cv >= +0,005 con mejora en >= 3/5 pliegues, o complementariedad demostrada (`ensemble_gain` >= 0,0015 con IC bootstrap > 0). **Bootstrap por tweet** (2.000 remuestreos) obligatorio para todo candidato.

### 26.2 Resultados por paquete (P0-P7)

| Paquete | Linea | Mejor variante | corrected_cv | ? vs V03 (0,7772) | Decision |
|---|---|---|---:|---:|---|
| P0 | Scaffold: holdout, OOF corregido, gap analysis | V03 reevaluada | 0,7772 |  -  | Referencia v2 |
| P1 | Cadenas de etiquetas (E2-01) | HS_AG_TR | 0,7744 | -0,0028 (1/5) | Cerrada (IC incluye 0) |
| P1 | Meta de dependencia de etiquetas (E2-02) | LR C=0,1 sobre 44 features | **0,7853** | **+0,0081** (4/5) | Adoptada como candidato (IC [+0,0021, +0,0248]) |
| P2 | Targeting estructural (E2-05/06) | union + n-gramas simbolicos | 0,7487 | -0,0285 (1/5) | Cerrada; recall de AG en la rebanada colectiva 0,452 -> 0,478 |
| P3 | POS / dependencias / morfologia sparse (E2-03/04) | union + POS | 0,7656 | -0,0116 (0/5) | Cerrada (dependencias 0,7653; morfologia 0,7567) |
| P4 | Cita/atribucion (E2-07) y negacion (E2-08) | union + negacion | 0,7657 | -0,0115 (0/5) | Cerrada (cita/atribucion 0,7554) |
| P5 | Emocion (E2-09) | union + afecto | 0,6330 | -0,1442 (0/5) | Cerrada |
| P5 | NB-SVM (E2-15) | NB-union + LR | 0,7612 | -0,0160 (2/5) | Cerrada (gain +0,0038, IC [-0,0081, +0,0162]) |
| P5 | Multicanal ortografico (E2-10) | raw + colapsado | 0,7534 | -0,0238 (0/5) | Cerrada |
| P6 | Stacking v2 (E2-17/18) | META-BASES-RIDGE alpha=3 | **0,7874** | **+0,0102** (3/5) | Adoptado por complementariedad (gain +0,0089, IC [+0,0019, +0,0160]); los canales no aportan (? -0,0006); recall de AG colectiva 0,452 -> 0,574 |
| P7 | Holdout congelado | meta Ridge alpha=3 | 0,8465 (V03 0,8285) | +0,0180 pareado | GO (IC [+0,0003, +0,0372]) |
| P7 | Sonda de leaderboard | v08 meta v2 |  -  | LB 0,78163 | **No adoptada** (-0,0014 vs v03) |
| v09 C-EMO | V03 + emocion (Ridge ?=0,3) | 0,78094 | 0,77895 | -0,0020 |
| v10 C-SYN | V03 + sintaxis (Ridge ?=1) | 0,77931 | 0,77978 | +0,0005 |
| v11 C-EMO-SYN | V03 + emocion + sintaxis (Ridge ?=1) | 0,77966 | 0,78150 | +0,0018 |

Lectura: las lineas de representacion (targeting, sintaxis, cita/negacion, emocion, NB-SVM, ortografia) se cerraron sin mejora, repitiendo el patron historico de que la concatenacion de canales degrada; el unico valor aparecio en la **fusion tardia a nivel meta**, que mejoro en desarrollo y en el holdout congelado pero no en el test.

### 26.3 El candidato y su validacion

El candidato v2 es el **meta-modelo de dependencia de etiquetas** (RidgeClassifier alpha=3, balanced, sobre 44 meta-features: probabilidades, margenes y entropias OOF de los 4 bases de texto mas 8 rasgos estructurales), adoptado en P6 por complementariedad.

- **Desarrollo**: corrected_cv **0,7874** (+0,0102 vs V03; 3/5 pliegues). Su ventaja de nivel frente al meta de P1 (0,7853) es ruido (IC [-0,0054, +0,0093] incluye 0); lo que lo respalda es la complementariedad (`ensemble_gain` +0,0089, IC [+0,0019, +0,0160]).
- **Holdout congelado** (248 tweets, 2/5 evaluaciones usadas): V03 **0,8285** (HS 0,857 / TR 0,849 / AG 0,779) frente al meta **0,8465** (HS 0,862 / TR 0,857 / AG 0,820); delta pareado **+0,0180** con IC95 [+0,0003, +0,0372]. El holdout resulto un pliegue mas facil que cualquier pliegue del desarrollo (0,83/0,85 vs 0,774/0,787).
- **Auditoria (`ml-reviewer`)**: reprodujo los numeros exactos, verifico 0 grupos/textos compartidos con el desarrollo, confirmo que no hay fuga en el stacking y que la submission v08 coincide con el pipeline declarado.
- **Sonda de leaderboard v08 = 0,78163** (positivos predichos HS 44,8 % / TR 32,5 % / AG 38,6 %), por debajo de v03 (0,78306) en -0,0014. El meta mejoro en desarrollo (+0,0102) y en el holdout congelado (+0,0180) pero **no transfirio al test**; su gap dev-LB fue -0,0058.

### 26.4 Lectura honesta del leaderboard

La fase v2 y la fase final de candidatos dejaron **12 sondas** en `submissions/leaderboard_log.csv`:

| Sonda | Modelo | Local (OOF) | Kaggle | Gap |
|---|---|---:|---:|---:|
| v00 plantilla | Todo 1 |  -  | 0,51494 |  -  |
| v01 baseline | TF-IDF word + LR | 0,75778 | 0,75677 | -0,0010 |
| v02 union u=0,5 | MOD-03b, umbral 0,5 | 0,79270 | 0,77993 | -0,0128 |
| v03 union + umbrales | MOD-03b + 0,44/0,51/0,46 | 0,79890 | 0,78306 | -0,0158 |
| v04 ensemble mediana | ENS-04 | 0,79850 | 0,77510 | -0,0234 |
| v05 optuna | OPT-01 + umbrales | 0,80390 | 0,77602 | -0,0279 |
| v06 robustez C=1.0 | MOD-03b C=1.0 + umbrales | 0,79290 | 0,77227 | -0,0206 |
| v07 robustez min_df=5 | min_df=5, C=1.0 + umbrales | 0,79930 | 0,77034 | -0,0290 |
| v08 meta v2 | Meta Ridge alpha=3 (stacking v2) | 0,78740 | 0,78163 | -0,0058 |

- **La media de las 7 sondas de la familia union (v02-v08) es 0,7769** y el rango completo de esas sondas (0,77034-0,78306) es de 0,0127, es decir **? 0,99 errores estandar** (SE ? 0,0129 con n = 1.154): el ranking del leaderboard a este nivel es, en gran parte, ruido de muestreo.
- El maximo observado (0,78306 de v03) esta **sesgado hacia arriba por seleccion (winner's curse)**: se eligio como maximo de 12 mediciones ruidosas, y tratarlo como objetivo reproducible seria un error.
- Los unicos contrastes pareados fiables son los que comparten casi todo el modelo: **v02 vs v03 (+0,0031** por umbrales) y **v03 vs v08 (-0,0014)**; ambos muy por debajo del error estandar.
- La sonda v08 confirma que la ventaja local del meta (respaldada incluso por un holdout congelado) no predijo la transferencia; la unica lectura que resiste es que **V03 y el meta son indistinguibles en el test**.
- Las tres sondas de la fase final (v09-v11, metas restringidos sobre V03 + canales) obtuvieron 0,77895 / 0,77978 / 0,78150: las tres por debajo de V03 y dentro de la banda; el ranking local volvio a no predecir el del LB (C-EMO era el mejor local de los tres y obtuvo el peor LB; C-EMO-SYN, el segundo local, obtuvo el mejor).

Nota de comparabilidad: el local de v08 es el `corrected_cv` del desarrollo (1.483 filas, protocolo D-016) y no es directamente comparable con el OOF historico de v01-v07 (1.731 filas, solo near-duplicados); el gap de v08 se calcula contra ese local corregido, igual que el +0,0059 implicito de v03 en la misma base (0,78306 - 0,7772).

### 26.5 Hallazgos del corpus en la fase v2

El analisis de los canales de cita/atribucion y negacion produjo una caracterizacion nueva del corpus sobre el desarrollo: solo el **2 %** de los tweets contiene comillas, hay **0 RT**, los marcadores de atribucion (segun/dijo/afirmo y similares) aparecen en el **6,9 %** y la negacion en el **25,5 %**. Hallazgo de herramienta: `es_core_news_md` **no emite** la etiqueta de dependencia `dep_=neg`; la negacion se detecto combinando `advmod` con `Polarity=Neg`.

### 26.6 Hipotesis abiertas

1. **Cambio de calibracion de las entradas del meta**: el meta se entrena con scores OOF de bases entrenados en 4/5 del desarrollo, pero en la sonda predice el test con bases entrenados con todo el train; ese cambio de calibracion de las entradas no se midio y es una hipotesis abierta. Se suma el gap dev-LB negativo de la familia union (-0,013 a -0,016).
2. **Backlog v2 cerrado al 100 % (24 de 24 lineas)**: el paquete P8 evaluo las cinco lineas restantes: calibracion Platt/isotonica anidadas (mejora Brier 0.1583 -> 0.1199 y log-loss 0.5002 -> 0.4035, sin cambio de F1 con umbrales reajustados); encoder de entidades por mencion/hashtag/dominio (cobertura informativa 0.5 %, cerrada por sparsity); umbrales por rebanada con shrinkage (curva monotona, el umbral local sobreajusta, -0.002 frente al global anidado); especialistas por rebanada (11 de 21 celdas significativas tras Benjamini-Hochberg, pero rinden de media -0.045 dentro de su rebanada); y espacios de features por etiqueta (-0.0253, 0/5 pliegues). Ninguna supera la regla de adopcion (D-016).

## Referencias

Las 45 referencias completas del estado del arte estan en `references.bib` y `reports/estado_arte.csv`. Se listan aqui las citadas en este informe, con DOI/URL verificados:

1. Alvarez-Carmona, M. A., et al. (2018). Overview of MEX-A3T at IberEval 2018: Authorship and Aggressiveness Analysis in Mexican Spanish Tweets. CEUR-WS Vol-2150. https://ceur-ws.org/Vol-2150/overview-mex-a3t.pdf
2. Aragon, M. E., et al. (2019). Overview of MEX-A3T at IberLEF 2019. CEUR-WS Vol-2421. https://ceur-ws.org/Vol-2421/MEX-A3T_overview.pdf
3. Arango, A., Perez, J., y Poblete, B. (2019). Hate Speech Detection is Not as Easy as You May Think: A Closer Look at Model Validation. SIGIR 2019, 45-54. DOI 10.1145/3331184.3331262
4. Argota Vega, L. E., Reyes-Magana, J. C., Gomez-Adorno, H., y Bel-Enguix, G. (2019). MineriaUNAM at SemEval-2019 Task 5: Detecting Hate Speech in Twitter using Multiple Features in a Combinatorial Framework. SemEval-2019, 447-452. DOI 10.18653/v1/S19-2079
5. Basile, V., Bosco, C., Fersini, E., Nozza, D., Patti, V., Rangel Pardo, F. M., Rosso, P., y Sanguinetti, M. (2019). SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter. SemEval-2019, 54-63. DOI 10.18653/v1/S19-2007
6. Bassignana, E., Basile, V., y Patti, V. (2018). HurtLex: A Multilingual Lexicon of Words to Hurt. CLiC-it 2018, CEUR-WS Vol-2253. DOI 10.4000/books.aaccademia.3085
7. Buda, M., Maki, A., y Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. Neural Networks, 106, 249-259. DOI 10.1016/j.neunet.2018.07.011
8. Davidson, T., Warmsley, D., Macy, M., y Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. ICWSM 2017, 512-515. DOI 10.1609/icwsm.v11i1.14955
9. Davidson, T., Bhattacharya, D., y Weber, I. (2019). Racial Bias in Hate Speech and Abusive Language Detection Datasets. ALW 2019, 25-35. DOI 10.18653/v1/W19-3504
10. ElSherief, M., Kulkarni, V., Nguyen, D., Wang, W. Y., y Belding, E. (2018). Hate Lingo: A Target-Based Linguistic Analysis of Hate Speech in Social Media. ICWSM 2018, 42-51. DOI 10.1609/icwsm.v12i1.15041
11. ElSherief, M., Ziems, C., Muchlinski, D., Gupta, V., Kambhatla, S., y Yang, W. (2021). Latent Hatred: A Benchmark for Understanding Implicit Hate Speech. EMNLP 2021, 345-363. DOI 10.18653/v1/2021.emnlp-main.29
12. Fortuna, P., y Nunes, S. (2018). A Survey on Automatic Detection of Hate Speech in Text. ACM Computing Surveys, 51(4), 85:1-85:30. DOI 10.1145/3232676
13. Founta, A.-M., et al. (2018). Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior. ICWSM 2018. DOI 10.1609/icwsm.v12i1.14991
14. Garcia-Diaz, J. A., Canovas-Garcia, M., Colomo-Palacios, R., y Valencia-Garcia, R. (2021). Detecting misogyny in Spanish tweets. An approach based on linguistics features and word embeddings. Future Generation Computer Systems, 114, 514-524. DOI 10.1016/j.future.2020.08.032
15. Garcia-Diaz, J. A., Jimenez-Zafra, S. M., Garcia-Cumbreras, M. A., y Valencia-Garcia, R. (2022). Evaluating feature combination strategies for hate-speech detection in Spanish using linguistic features and transformers. Complex & Intelligent Systems, 8, 4901-4916. DOI 10.1007/s40747-022-00693-x
16. Garibo i Orts, O. (2019). Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter at SemEval-2019 Task 5: Frequency Analysis Interpolation for Hate in Speech Detection. SemEval-2019, 460-463. DOI 10.18653/v1/S19-2081
17. Jahan, M. S., y Oussalah, M. (2023). A systematic review of hate speech automatic detection using natural language processing. Neurocomputing, 546, 126232. DOI 10.1016/j.neucom.2023.126232
18. Joachims, T. (1998). Text Categorization with Support Vector Machines: Learning with Many Relevant Features. ECML-98, 137-142. DOI 10.1007/BFb0026683
19. Lipton, Z. C., Elkan, C., y Naryanaswamy, B. (2014). Optimal Thresholding of Classifiers to Maximize F1 Measure. ECML PKDD 2014, 225-239. DOI 10.1007/978-3-662-44851-9_15
20. MacAvaney, S., Yao, H.-R., Yang, E., Russell, K., Goharian, N., y Frieder, O. (2019). Hate speech detection: Challenges and solutions. PLOS ONE, 14(8), e0221152. DOI 10.1371/journal.pone.0221152
21. Malmasi, S., y Zampieri, M. (2017). Detecting Hate Speech in Social Media. RANLP 2017, 467-472. DOI 10.26615/978-954-452-049-6_062
22. Mishra, P., Del Tredici, M., Yannakoudakis, H., y Shutova, E. (2018). Neural Character-based Composition Models for Abuse Detection. ALW 2018. DOI 10.18653/v1/W18-5101
23. Nan, Y., Chai, K. M. A., Lee, W. S., y Chieu, H. L. (2012). Optimizing F-measure: A Tale of Two Approaches. ICML 2012. DOI 10.48550/arXiv.1206.4625
24. Nina-Alcocer, V. (2019). HATERecognizer at SemEval-2019 Task 5: Using Features and Neural Networks to Face Hate Recognition. SemEval-2019, 409-415. DOI 10.18653/v1/S19-2072
25. Nobata, C., Tetreault, J., Thomas, A., Mehdad, Y., y Chang, Y. (2016). Abusive Language Detection in Online User Content. WWW 2016, 145-153. DOI 10.1145/2872427.2883062
26. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
27. Pereira-Kohatsu, J. C., Quijano-Sanchez, L., Liberatore, F., y Camacho-Collados, M. (2019). Detecting and Monitoring Hate Speech in Twitter. Sensors, 19(21), 4654. DOI 10.3390/s19214654
28. Plaza-del-Arco, F. M., Molina-Gonzalez, M. D., Martin, M., y Urena-Lopez, L. A. (2019). SINAI at SemEval-2019 Task 5: Ensemble learning to detect hate speech against inmigrants and women in English and Spanish tweets. SemEval-2019, 476-479. DOI 10.18653/v1/S19-2084
29. Plaza-del-Arco, F. M., Molina-Gonzalez, M. D., Urena-Lopez, L. A., y Martin-Valdivia, M. T. (2021). Comparing pre-trained language models for Spanish hate speech detection. Expert Systems with Applications, 166, 114120. DOI 10.1016/j.eswa.2020.114120
30. Plaza-del-Arco, F. M., Halterman, A., Weerasooriya, T., y Dredze, M. (2021). A Multi-Task Learning Approach to Hate Speech Detection Leveraging Sentiment Analysis. IEEE Access, 9. DOI 10.1109/ACCESS.2021.3103697
31. Poletto, F., Basile, V., Sanguinetti, M., y Bosco, C. (2021). Resources and benchmark corpora for hate speech detection: a systematic review. Language Resources and Evaluation, 55, 477-523. DOI 10.1007/s10579-020-09502-8
32. Rottger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H., y Pierrehumbert, J. (2021). HateCheck: Functional Tests for Hate Speech Detection Models. ACL-IJCNLP 2021, 41-58. DOI 10.18653/v1/2021.acl-long.4
33. Schmidt, A., y Wiegand, M. (2017). A Survey on Hate Speech Detection using Natural Language Processing. SocialNLP 2017, 1-10. DOI 10.18653/v1/W17-1101
34. Silla, C. N., y Freitas, A. A. (2011). A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22, 31-72. DOI 10.1007/s10618-010-0175-9
35. Stamatatos, E. (2009). A survey of modern authorship attribution methods. JASIST, 60(3), 538-556. DOI 10.1002/asi.21001
36. Stamatatos, E. (2017). Authorship Attribution Using Text Distortion. EACL 2017, 1138-1149. DOI 10.18653/v1/E17-1107
37. Vidgen, B., y Derczynski, L. (2020). Directions in abusive language training data, a systematic review: Garbage in, garbage out. PLOS ONE, 15(12), e0243300. DOI 10.1371/journal.pone.0243300
38. Vidgen, B., Harris, A., Nguyen, D., Tromble, R., Hale, S., y Margetts, H. (2019). Challenges and frontiers in abusive content detection. ALW 2019, 80-93. DOI 10.18653/v1/W19-3509
39. Wiegand, M., Ruppenhofer, J., Schmidt, A., y Greenberg, C. (2018). Inducing a Lexicon of Abusive Words  -  a Feature-Based Approach. NAACL-HLT 2018, 1046-1056. DOI 10.18653/v1/N18-1095
40. Wiegand, M., Ruppenhofer, J., y Kleinbauer, T. (2019). Detection of Abusive Language: the Problem of Biased Datasets. NAACL-HLT 2019, 581-588. DOI 10.18653/v1/N19-1060
41. Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241-259. DOI 10.1016/S0893-6080(05)80023-1
42. Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., y Kumar, R. (2019). Predicting the Type and Target of Offensive Posts in Social Media. NAACL-HLT 2019, 1415-1420. DOI 10.18653/v1/N19-1144

---

### Apendice: artefactos de evidencia citados

- Gobernanza: `PROJECT_STATE.md`, `DECISIONS.md` (D-001 a D-015), `RUNLOG.md`, `experiments.csv` (51 experimentos).
- Datos y auditoria: `reports/auditoria_resumen.json`, `reports/data_dictionary.csv`, `reports/leakage_audit.md`, `data/processed/grupos_near_duplicados.csv`.
- Estado del arte e hipotesis: `reports/estado_arte.md`, `references.bib`, `reports/hipotesis.md`.
- Analisis: `reports/eda_resumen.md` (43 artefactos en `reports/eda/`), `reports/error_analysis.md`, 26 figuras en `reports/figuras/`.
- Resultados: `reports/variantes_representaciones.csv`, `reports/features_densas_resultados.csv`, `reports/ensembles_resultados.csv`, `reports/optuna_trials.csv`, `reports/dl_desde_cero_resultados.csv`, `reports/control_pseudo_autores.json`, `reports/fase13a_refit_umbral_metricas.json`, `reports/fase13d_sondas_robustez.json`.
- Competencia: `submissions/leaderboard_log.csv`, 8 submissions en `submissions/`, reglas en `docs/competition_rules.md`.
- Auditoria final y reproducibilidad: `reports/auditoria_final.md` (veredicto GO; scripts de verificacion en `/tmp/opencode/` y copia aislada en `/tmp/opencode/repro_kaggle_toxic/`).

---

## 27. Investigacion posterior (fase v4 "Advanced Research v4")

Esta seccion documenta una fase de investigacion posterior a la entrega, abierta
el 2026-09-25 con una especificacion de 108 secciones
(`docs/advanced_research_v4_spec.md`) bajo la pregunta "que propiedad de los
datos, de las etiquetas, de la representacion o del proceso de aprendizaje
todavia no estamos explotando?". No modifica el modelo final: su valor es
acotar con evidencia que parte del error es ruido de anotacion, que parte es
dificultad del modelo y que parte es limite de la representacion.

### 27.1 Protocolo

Se mantiene integro el protocolo v2 (D-016): grupos = near-duplicado ?
pseudo-autor, desarrollo de 1.483 tweets, holdout congelado de 248, `corrected_cv`
como metrica primaria y bootstrap pareado por tweet (2.000 remuestreos). La
referencia V03 se verifico byte a byte al abrir la fase (5/5 hashes, `corrected_cv`
0.777211 reproducida exactamente) y de nuevo en la auditoria final (veredicto GO,
0 violaciones de reglas en 12 scripts). Las 26 lineas del backlog se
pre-registraron antes de experimentar para controlar la multiplicidad.

### 27.2 Resultados por track

| Track | Resultado | Cifra clave |
|---|---|---|
| A. Integridad de datos y ruido de etiqueta | Cerrado | 15 clusters de texto repetido (84 tweets, 4.85 %), solo **2 contradictorios**; el error OOF es **menor** dentro (HS 0.046 vs 0.218) |
| B. Dureza de ejemplos y reweighting | Cerrado | El hard example mining **empeora** (-0.0101 con IC negativo); bajar el peso de los que mas violan el umbral **mejora** (+0.0036 en LR, +0.0086 en SVC) |
| C. Weighting de la representacion | Cerrado | log-odds 0.7571, chi2 0.6622, MI 0.7378, BM25 0.7748-0.7755, binario 0.7738: **todas degradan** frente a 0.7772 |
| D. Representaciones sparse nuevas | Cerrado | Afijos 0.7758, interacciones y ventanas por debajo: el canal de morfologia (brecha FASE 13) queda cerrado |
| E. Modelo conjunto de etiquetas | Cerrado | Label Powerset sobre 6 y 4 combinaciones: mejor 0.7713, **no supera** a los tres binarios independientes |
| F. Robustez y shift | Cerrado | 2 de 10 variables no supervisadas con shift; V03 da **0.7568 en textos unicos** y **0.9769 en los 68 near-duplicados**; la propagacion por near-duplicado tiene techo nulo |
| G. Modelos estadisticos alternativos | Mixto | La verosimilitud class-conditional aporta **+0.0041** en fusion tardia; el **SVC RBF sobre SVD 300 llega a 0.7904** con IC fragil |
| H. Fusion y calibracion del meta | Cerrado | El shift entre regimenes de entrenamiento es pequeno (KS 0.062) y **no causa dano**; se refuta la hipotesis abierta de la fase v2 |

### 27.3 Tres resultados que cambian la lectura del problema

1. **El error no es ruido de anotacion.** La contradiccion textual es un
   fenomeno escaso (2.8 % de los ejemplos) y concentrado en un unico cluster de
   48 tweets con el mismo patron de insulto. El modelo clasifica bien los
   ejemplos ambiguos, y las tres estrategias de la especificacion (baseline,
   downweight, eliminacion de contradicciones extremas) mueven la F1 en +/-0.0005.
2. **La forma de medir la dureza importa, y la propuesta por la BCE no sirve en
   este corpus.** La dificultad por etiqueta ordena bien el error (r ? -0.80),
   pero **agregar las tres etiquetas rompe el orden** (r = -0.136) por
   compensacion: los ejemplos que fallan mas tienen la BCE media *menor* (0.58
   frente a 1.34). La distancia a la frontera de decision si es monotona
   (r = +0.345) y es la que se uso.
3. **La mejora local no es un predictor de lageneralizacion.** Por primera vez se
   tenia un candidato que superaba los tres criterios de adopcion *y* una
   medicion externa limpia: `meta36_downweight` (corrected_cv 0.7882 = +0.0110,
   4/5 pliegues, IC95 del delta pareado [+0.0054, +0.0200], correccion neta +34
   frente a V03). El holdout congelado, que no habia participado en ninguna
   decision de la fase, lo rechazo: **0.8163 frente a 0.8285 de V03 (delta
   -0.0122)**.

### 27.4 Decision y efectos sobre la entrega

- **Ningun candidato de la fase v4 se adopta.** V03 sigue siendo el modelo y la
  submission finales; no se genero ninguna sonda nueva de leaderboard (siguen
  siendo 12) y la submission v12 se conserva como artefacto reproducible
  marcado `DESCARTADO` en `submissions/submission_manifest.csv`.
- El holdout queda agotado (5/5 evaluaciones) y su valor queda registrado en el
  registro: es la unica particion que no participo en ninguna decision y es la
  que detecta la falta de transferencia.
- Se anade a `experiments.csv` un total de **15 experimentos nuevos** (143 filas) y
  cinco decisiones (D-025 a D-029), con la misma trazabilidad que las fases
  anteriores.
- Los entregables se actualizan sin alterar lo ya validado: el notebook incorpora
  la seccion 17 (44 celdas, 0 errores, 36 comprobaciones [OK]) y este informe
  incorpora la seccion 27; los conteos verificados por el notebook pasan a 143
  experimentos, 39 scripts y 12 sondas.

### 27.5 Lo que la fase deja como evidencia reutilizable

- La contradiccion de etiquetas se puede cuantificar de forma barata y
  automatizable (cuatro criterios de agrupamiento, 3 s de computo), y sirve para
  descartar de entrada toda la familia de "reparar el ruido de anotacion".
- Cualquier reweighting futuro en este proyecto debe usar la distancia a la
  frontera o la violacion del umbral, con signo **negativo** sobre los ejemplos
  que el modelo falla, y debe validarse con el holdout antes de cualquier sonda.
- La descomposicion del desempeno por subconjunto (0.7568 en textos unicos frente
  a 0.9769 en near-duplicados) es la explicacion mas directa de por que la F1
  agregada es una metrica fragil en un corpus con repeticion.

## 28. Investigacion posterior  -  Embedding From Scratch (v5 y v5.1)

Esta seccion documenta la ultima fase del proyecto: **aprender una representacion
vectorial de las palabras desde cero**, sin transformers ni embeddings preentrenados, y
probar si aporta informacion que el modelo final V03 no tenga ya. Cubre dos fases:
**v5** (?de donde salen los datos y que pasa al cambiar de corpus?) y **v5.1** (?alguna
configuracion de la arquitectura aporta mas?).

### 28.1 Motivacion y resultado en una linea

V03 funciona porque lee la **superficie** del texto: el insulto exacto, la mencion, la
deformacion ortografica. Lo que no tiene es una nocion de **similitud**, y con 1.483
ejemplos etiquetados no hay forma de aprenderla. La hipotesis era que una representacion
distributiva anadiria esa dimension. **El resultado es negativo y esta documentado como
tal**: la representacion mejora de forma medible, pero no aporta a la tarea, y **V03 se
mantiene** en `corrected_cv` **0.777211** (LB **0.78306**).

### 28.2 Corpus: construccion y autorizacion

| corpus | tokens | documentos | licencia | papel |
|---|---|---|---|---|
| Wikipedia ES 5M | 5.011.404 | 1.694 | CC BY-SA 4.0 | tramo inferior de la curva |
| Wikipedia ES 10M | 10.002.400 | 2.847 | CC BY-SA 4.0 | corpus de referencia |
| Wikipedia ES 50M | 50.000.028 | 21.529 | CC BY-SA 4.0 | tramo superior |
| mC4-es 10M | 10.000.239 | 16.738 | ODC-BY 1.0 | registro web |
| mixto 20M | 20.000.045 |  -  | union de los dos anteriores | efecto del registro |

Pipeline de siete pasos, **identico para todos** para que la comparacion sea de cantidad de
datos y no de receta: `download` -> `raw` -> `clean` -> `dedup` -> `normalize` -> `tokenize` ->
`shards` + `manifest`. La normalizacion es la variante **EMB-NORMAL**: minusculas
conservando tildes, URLs, menciones, hashtags y numeros reducidos a marcadores estables,
**sin corrector ortografico y sin reconstruccion de tildes**.

La **autorizacion** (D-030) permite corpus externos **sin etiquetar solo para aprender la
representacion** y prohibe todo uso supervisado. Se registra tal como se comunico, sin
documentacion escrita inventada. OSCAR, que era *gated*, no se forzo.

**Detalle de trazabilidad que resulto ser una correccion**: el manifiesto guardaba el
`sha256` del **volcado**, que no identifica la escala porque 5M, 10M y 50M de Wikipedia
salen del mismo tramo. La identidad de un entrenamiento usa el `sha256` del corpus **ya
procesado**, que si distingue las tres escalas (D-035).

### 28.3 Cobertura y OOV

La cobertura se mide contra el vocabulario TOXIC de train y test (uso como diagnostico,
autorizado; no como senal).

| corpus | por tipo | por token |
|---|---|---|
| Wikipedia 5M | 0.6302 | 0.8646 |
| Wikipedia 10M | 0.6839 | 0.8833 |
| Wikipedia 50M | 0.7703 | 0.9238 |
| web mC4-es 10M | 0.8256 | 0.9512 |
| mixto 20M | 0.8413 | 0.9553 |

**Hallazgo central de la fase**: la cobertura y la aportacion **se mueven por separado**. El
corpus web sube la cobertura del 0,68 al 0,83 y la aportacion a V03 cambia en milesimas. Y la
hipotesis de que faltara senal por "no tener tildes" queda refutada: el 8,5 % de los tipos
ausentes si lo estan, y el 66,6 % son **deformaciones con perdida de vocales**.

### 28.4 FastText y el control Word2Vec

El control es el mismo binario con las subpalabras apagadas, que es Word2Vec. La
comparacion aisla **una sola cosa**:

| modelo | tipos con vector de 6.059 | fusion |
|---|---|---|
| FastText 200d (subpalabras 3-6) | **6.059** | 0.7818 |
| Word2Vec-control (10M) | 5.134 | 0.7762 |
| FastText 200d (50M) | **6.059** | 0.7803 |
| Word2Vec-control (50M) | 4.412 | 0.7773 |

Los 925 tipos que el control pierde son formas ortograficas, y son las que deciden la
etiqueta en un corpus de tuits. La ventaja de las subpalabras tiene forma de **joroba** con
maximo en 10M, no una relacion monotona con la escala (D-034).

### 28.5 Downstream y la regla de escala

| pooling | corrected_cv |
|---|---|
| media | 0.6603 |
| media + maximo | 0.6388 |
| ponderado por IDF | 0.6348 |
| **suelo aleatorio** | **0.5404** |

La regla de escala importa mas que el pooling: al fusionar, el bloque denso se normaliza
**L2 por documento**. Estandarizarlo hace que las 200 dimensiones dominen al bloque disperso
y la fusion **pierda 0,09 de F1** (0.7818 con L2 frente a 0.6901) sin que eso diga nada
sobre la representacion: son los mismos datos.

### 28.6 Fusion y escala del corpus

Cinco configuraciones, y el delta de aportacion vive siempre entre **+0.0001 y +0.0046**,
con el intervalo de bootstrap tocando el 0:

| corpus | fusion | delta |
|---|---|---|
| Wikipedia 5M | 0.7757 | **-0.0015** |
| Wikipedia 10M | 0.7818 | +0.0046 |
| Wikipedia 50M | 0.7803 | +0.0031 |
| web mC4-es 10M | 0.7812 | +0.0040 |
| mixto 20M | 0.7776 | +0.0004 |

**Conclusion de la curva**: multiplicar por diez los tokens (5M -> 50M) **reduce** el delta.
El techo no es la representacion, y **no se escala el corpus**: tres de las cinco condiciones
de la spec fallan, y las que fallan son de evidencia, no de coste (D-041).

### 28.7 Arquitectura (v5.1): las cinco familias

Todas sobre Wikipedia 10M, cambiando **un parametro** cada vez. Criterio de adopcion de dos
niveles: **delta >= +0,005 Y el intervalo de confianza no incluye el 0** (D-036).

| familia | mejor fusion | delta | IC95 | veredicto |
|---|---|---|---|---|
| vector_size | 0.7823 (100d) | +0.0050 | [-0.0028, +0.0130] | cerrada (inerte) |
| window | 0.7830 (ws 3) | +0.0057 | [-0.0021, +0.0136] | cerrada (inerte) |
| min_count | 0.7818 (baseline) | +0.0046 | [-0.0032, +0.0124] | cerrada (inerte) |
| subwords | 0.7818 (baseline) | +0.0046 | [-0.0032, +0.0124] | cerrada (inerte) |
| negative | 0.7818 (baseline) | +0.0046 | [-0.0032, +0.0124] | cerrada (inerte) |

**14 configuraciones, 0 adoptables.** Tres lecturas que quedan como resultado:

- **La dimension no es la palanca**: con 1.483 ejemplos, 100, 200 y 300 distancias dan lo
  mismo (rango 0.0017).
- **Menos ventana rinde mas** (3 > 5 > 7), y el intervalo de subpalabras se queda en el
  punto medio 3-6: acortar o alargar empeora. La senal util es **local y ortografica**.
- **`min_count` no puede funcionar con subpalabras**: bajar el umbral agranda el vocabulario
  general pero no anade ni un tipo de TOXIC con vector, porque fastText ya compone vector
  para cualquier tipo consultation. Su via de mejora ya esta ocupada.

### 28.8 Complementariedad y resultados negativos

- **Que arregla**: el embedding aporta **la misma informacion que V03 con mas ruido**. Su
  variante con mejor delta (WS-3) no arregla mas filas que el baseline (8 frente a 8):
  arregla lo mismo y **rompe menos** (2 frente a 5). Toda la ventaja esta en el *damage*.
- **Rangos**: el embedding aislado se mueve en 0,0297 entre 0,6397 y 0,6694; la fusion se
  mueve en 0,0063. El rango de la representacion es **4,7 veces mayor** que el de la tarea.
- **Resultados negativos explicitos**: 100d y 300d (no superan el baseline), ventanas 3 y 7
  (mejoran el embedding pero no la fusion), min_count 2 y 5 (ninguno supera el 3), subpalabras
  3-5 y 3-7 (ninguno supera el 3-6), negative 5 (cae por debajo del 10), mezcla de corpus
  (la peor: +0.0004), estandarizar el bloque denso (-0.0871), los cinco corpora por debajo
  del gate, y el meta-modelo anidado de v5 (-0.0094).

### 28.9 Contabilidad de la fase

- 22 scripts de la linea del embedding y 138
  evaluaciones downstream registradas, con doble contabilidad (entrenamientos = evaluaciones).
- Presupuesto: **9 de 9**
  exploratorios y **0 de 1** de
  reserva. La reserva queda **intacta** porque su unico fin era confirmar una mejora ya
  vista, y no la hay.
- 42 decisiones registradas en total; esta linea ocupa D-030 a D-042.

### 28.10 Conclusion

La fase entrega dos conclusiones separadas y las mantiene separadas a proposito, porque en
v5 se confundieron una vez:

1. **Representacion**: la arquitectura **si** cambia la calidad del embedding (rango
   0,0297) y su forma revela que la senal util es local y ortografica. Es un resultado
   positivo como propiedad del embedding.
2. **Tarea**: ninguna configuracion aporta a V03 por encima del ruido. El mejor delta de la
   tabla es +0,0057 con IC95 [-0,0021, +0,0136]. Es un resultado negativo, y ninguno anula
   al otro.

**V03 sigue siendo el modelo final.** La unica variable que la evidencia deja en pie es el
**tamano de los datos etiquetados** (1.483): con mas etiquetas, tanto el bloque disperso
como el denso podrian aportar mas de lo que aportan hoy.

*Seccion generada por `experiments/embedding_21_documentacion_v5_1.py` (EBD-27 (P9)) el
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


---

## Addendum P49 (2026-09-28) - La contradiccion del rescate, resuelta

### Que significan las tres cifras

| cifra | unidad | que es |
|---|---|---|
| **757** | label-cell | celdas (tweet, etiqueta) con `pred != y_true` sobre 1483 x 3. Los tweets con error son **441** |
| **147** | label-cell | celdas de error de V03 que acierta **una** variante, `C_meta36_downweight` con umbral 0.5 |
| **715** | label-cell | celdas de error de V03 que acierta **al menos uno** de los 20 canales de la fase v2 |

**Las dos no son comparables.** 147 es el rescate de un modelo; 715 es el de la union de veinte.
La confusion venia de ponerlas sobre el mismo denominador sin decir que los numeradores
significan cosas distintas.

### Los tensores no se habian perdido

Estaban en `/home/carlos/Git/PLN/.../features/` y en `/home/carlos/Git/PLN-v8/.../features/`.
Reconstruido el pool de 20 canales con la misma regla de exclusion que
`experiments/28_hardness_v4.py`, la fraccion de fallo por celda coincide con
`reports/hardness_v4.csv` en **4449 de 4449 celdas con diferencia 0.00e+00**. La limitacion que
P48 declaro era falsa.

Y el 147 se reproduce exactamente: **147 rescates, 90 danios, cobertura 0.194187582562748**.

### 715 es real, y no es azar

| magnitud | valor |
|---|---:|
| probabilidad de que un canal acierte una celda de error de V03 | 0,2756 |
| celdas irrecuperables **esperadas** si los canales fueran independientes | **1,20** |
| celdas irrecuperables **observadas** | **42** |

Si el rescate grande fuera un artefacto de tener veinte modelos, las irrecuperables serian unas
pocas. Son 42, **35 veces mas**. Los canales fallan juntos con V03, que es la firma de un hueco
de representacion compartido.

### Pero el rescate no se puede cobrar

| canal | rescate | damage | net |
|---|---:|---:|---:|
| `cita:NEG` | 337 | 2043 | **-1706** |
| `synt:SYN-POS` | 309 | 1014 | -705 |
| `emoc:EMO-LEX@C=1.0` | 306 | 899 | -593 |
| `base:M2_union_lr_c2` | 71 | 60 | **+11** |
| los otros 16 | | | negativos |

**19 de 20 canales tienen net negativo.** El unico positivo es `M2_union_lr_c2`, que es el
propio V03 con umbral 0.5 en lugar de 0,44/0,51/0,46: no es un modelo alternativo, es una
perturbacion de umbral del mismo modelo.

Esta es la conclusion que P48 no tenia y que cambia el sentido de su veredicto. No es que no haya
senal de rescate. **Hay 715 celdas rescued, son reales, y son irreducibles en la practica**,
porque ningun canal las convierte en mejora neta y no existe informacion por celda que diga cual
de los veinte escuchar.

### Efecto sobre las conclusiones de P48

| elemento de P48 | estado tras P49 |
|---|---|
| 757 celdas de error | **verificado** |
| 715 rescatables | **verificado, y es observacion directa, no cota** |
| 42 irrecuperables | **verificado** |
| analisis de margen | **verificado**, y ahora sobre rescate observado |
| "los tensores no estan en disco" | **superado**: estaban en dos worktrees |
| criterio de enrichment | **superado**: solo miraba rescate, no irrecuperabilidad |
| `EDA_RESEARCH_GAP = NONE` | **se mantiene**, por motivo mas fuerte |

Y las slices cambian de diagnostico al anadir el enriquecimiento en irrecuperables: `negacion`
tiene lift 1,921, es decir se concentra en errores **irrecuperables**, mientras `hashtags` tiene
lift 0,245 y se concentra en errores **rescatables**. P48 las declaraba todas indistinguibles
porque solo midio una de las dos direcciones.

### Lo que queda abierto

Ninguna hipotesis de modelo. Pero si una pregunta de teoria de la informacion que la evidencia
ahora permite formular con precision: **?por que los veinte canales aciertan 715 celdas
distintas, y ninguno de forma exploitable?** Eso no es un problema de modelo, es un problema de
identificacion: falta una senal que diga, por celda, cual de los veinte canales merece ser
escuchado. Ninguna de las familias probadas en V21-V48 attacka eso.

*Seccion generada por `experiments/v49_00_rescate_forense.py` y
`experiments/v49_01_integrar_notebook.py`. Sin push: la rama sigue local.*


---

## Addendum P50 (2026-09-28) - El techo es enorme, y no se puede cobrar

### La pregunta de fondo, respondida

> El problema historico no era falta de senal, sino incapacidad para identificar que canal es
> correcto en cada celda?

**Si, en el diagnostico. No, en el remedio.** Y esa distincion es el resultado de la fase.

### El techo del oracle

| magnitud | valor |
|---|---:|
| macro-F1 de V03 | 0,7772114525 |
| macro-F1 del oracle | 0,9870178614 |
| **delta** | **+0,2098064089** |
| celdas con alguna accion correcta | 4407 de 4449 |
| celdas sin ninguna accion correcta | 42 |

Por etiqueta: HS +0,231382, TR +0,170787, AG +0,227250. **69,9 veces** el margen de 0,003 que la
directiva exigia. La idea de que macro-F1 rindiese poca ganancia practica queda **refutada**.

### La captura

| selector | corrected_cv | delta | rescate | damage | net |
|---|---:|---:|---:|---:|---:|
| B0 V03 | 0,7772114525 | +0,0000000000 | 0 | 0 | 0 |
| B1 mayoria de los 20 canales | 0,3502747884 | -0,4269366641 | 361 | 1645 | -1284 |
| B2 canal mas fiable | 0,3616799365 | -0,4155315160 | 373 | 1655 | -1282 |
| B3 V03 salvo mayoria | 0,3502747884 | -0,4269366641 | 361 | 1645 | -1284 |
| **S1 selector cross-fitted** | **0,7657442265** | **-0,0114672260** | **48** | **67** | **-19** |

**0 de 5 pliegues positivos.** Y la cifra que lo cierra: la accion que elige S1 acierta **82,56 %**
de las celdas, mientras no cambiar nunca acierta **82,98 %**. El selector es **peor que abstenerse**.

### Por que falla

La regla es `argmax` de P(acierta), luego premia al canal mas **predecible**, no al mas util. Lo que
eligio de verdad:

| accion | veces | net real |
|---|---:|---:|
| KEEP_V03 | 4158 | 0 |
| `cita:NEG` | 109 | **-1706** |
| `synt:SYN-POS` | 81 | -705 |
| `emoc:NB-WORD-LR@C=1.0` | 30 | -46 |

**Todos los canales que eligio tienen net negativo**, y los dos mas elegidos son los dos peores del
pool. El AUC maximo (0,7237) lo tiene precisamente el canal con peor net.

Y la razon de fondo: **19 de los 20 canales tienen accuracy inferior a 0,5 sobre el conjunto de
desacuerdo**, es decir sobre las celdas donde un selector los usaria. Solo V03 con umbral 0.5 llega
a 0,5420. El danio se concentra entero en la frontera: el bin de margen `[0, 0,10)` aporta **net -22**,
`[0,10, 0,20)` aporta +3, y `[0,20, 1]` no cambia una sola celda.

### Decision

`CASE B`. `CONTEXTUAL_SELECTION = CLOSED`. S2 no se ejecuta por regla de la directiva. Sin bootstrap
ni segunda particion, porque el gate de 0,003 no se pasa.

### Lo que esto cambia en el diagnostico del techo de V03

Hasta aqui la explicacion era "probamos 15 familias y ninguna funciono", que no distingue entre las
tres causas posibles. Ahora se distinguen dos, y la evidencia se inclina claramente por una:

- **No es que falte senal.** Hay +0,2098 de macro-F1 disponible en el pool historico.
- **Es que la senal es difusa y no direccionable.** 715 celdas tienen una alternativa correcta,
  ninguna tiene una firma observable que diga cual, y los canales son individualmente peores que una
  moneda en el preciso conjunto donde habria que elegir.

Es una explicacion mucho mas solida del techo actual, y tambien mas dura: **no queda una via legal de
seleccion contextual que explorar**, porque la dificultad no es de modelo sino de informacion.

*Seccion generada por `experiments/v50_00_brecha_y_oracle.py`,
`experiments/v50_01_selector.py` y `experiments/v50_02_integrar_notebook.py`. Sin push: la rama sigue
local.*


---

## Addendum P51 (2026-09-28) - La senal si se identifica, pero el conjunto de cambios es una perdida

### La pregunta de fondo, respondida

> ?Los 715 rescates potenciales pueden convertirse en mejora real si el selector aprende
> a predecir el beneficio de sustituir V03, en lugar de intentar predecir que canal
> acertara?

**No. Y el motivo no es el que se suponia.**

### P50 se equivoco en el diagnostico

P50 concluyo que la informacion complementaria no era identificable. P51 lo mide y lo
refuta:

| prueba | resultado |
|---|---|
| AUC out-of-fold del target de utilidad | **0,5507 a 0,9249, media 0,8091** |
| AUC out-of-fold del target de correccion (P50) | 0,55 a 0,77 |
| permutacion del net de celdas | observado **-62**, nula **-115**, IC95 [-136, -93] |
| p de una cola, mejor que el azar | **0,0000** |

El selector es **53 celdas mejor que colocar los cambios al azar**. La informacion si
se identifica.

### La causa real: el conjunto de cambios tiene expectativa negativa

No hacer nada tiene net **0**. El mejor selector conocido tiene net **-62**. La permutacion
muestra que ese -62 es **53 celdas mejor** que la politica aleatoria, luego el selector
trabaja, solo que **el pool de cambios disponibles es una perdida en conjunto** y recuperar
su ganancia entera no alcanza para superar la inaccion.

Hay ademas un **gap de calibracion** enorme: el modelo esperaba que sus cambios valieran
**+0,056618** de macro-F1 y salieron a **-0,010159**.

### La asimetria de direccion, que es el hallazgo mecanico

| magnitud | valor |
|---|---:|
| equilibrio para **subir** de 0 a 1, sobre `P(y=1)` | **0,3892** |
| equilibrio para **bajar** de 1 a 0, sobre `P(y=1)` | **0,3892** |
| los dos coinciden | si, a 1,7e-16 |
| razon `FN->TP` sobre `FP->TN` | **1,57x** |
| tasa base al subir / elevacion necesaria | 17,86 pct / **2,18x** |
| tasa base al bajar / elevacion necesaria | 37,61 pct / **1,03x** |

**El equilibrio no es 0,5, y sale de la forma cerrada de V19-01, no de un grid.** Y al
medir el reparto por direccion, el signo resulta invertido respecto a lo que predecian las
magnitudes:

| direccion | cambios | rescate | damage | net |
|---|---:|---:|---:|---:|
| subir 0 a 1 | 209 | 71 | 138 | **-67** |
| bajar 1 a 0 | 95 | 50 | 45 | **+5** |

Bajar sale **neta positiva**, porque el selector discrimina mucho mejor al bajar: 52,6 pct
de aciertos frente al equilibrio de 38,92 pct. La variante restringida a esa direccion da
net +5, o sea positiva en celdas pero **insuficiente en macro-F1**: 5 celdas valen unos
0,0006 y hacen faltan 26,75 para 0,003.

### Un hecho estructural que explica por que S1-U se desploma

En celdas binarias, si el canal y V03 discrepan, la etiqueta verdadera coincide con
**exactamente una** de las dos predicciones. Luego **todo cambio de discrepancia es
rescate o dano, y nunca ambos mal**, y por tanto `delta_F1 > 0` **equivale a** rescate.
El target binario de P51 coincide por construccion con el de P50. Lo unico nuevo es la
magnitud, y un clasificador binario no puede usarla: trata igual una celda que gana
1,0e-3 que otra que gana 1,6e-3, de modo que no sabe que bajar de 1 a 0 arriesga
`TP -> FN`, que vale 1,57 veces mas de lo que gana. Por eso destroza el margen alto: 1018
cambios, net -780.

### Las tres politicas

| politica | corrected_cv | delta vs V03 | net |
|---|---:|---:|---:|
| V03 | 0,7772114525 | +0,0000000000 | 0 |
| P50 S1 (`channel_correct`) | 0,7657442265 | -0,0114672260 | -19 |
| P51 S1-U (binario) | 0,0294622851 | -0,7477491670 | -916 |
| P51 S2-U (utilidad esperada) | 0,7670528094 | -0,0101586431 | -62 |
| oracle (techo, usa `y_true`) | 0,9870178614 | +0,2098064089 | +715 |

### Decision

`CASE A`. `CONTEXTUAL_SELECTION = CLOSED` **de forma definitiva**, tras haber probado
gating por correccion y gating por utilidad, con dos formulaciones de selector cada uno.

La conclusion final sobre el techo de V03 es mas precisa que la de P50 y mas dura:

- **No es que falte senal.** Hay +0,2098 de macro-F1 disponible, y es identificable
  con AUC 0,81.
- **Es que el pool de cambios tiene expectativa negativa.** De media, cambiar V03 por
  un canal arbitrario pierde. El selector recupera 53 celdas de esa perdida y no
  suficientes para volver al positivo.

Eso explica por que 15 familias de modelos no,superaron el techo, y tambien por que ningun
selector contextual lo hara con este pool. Para convertirlo habria que anadir canales
con **net positivo individual**, y P49 ya midio que **no hay ninguno**: de 20 canales, 19
tienen net negativo y el unico positivo es el propio V03 con otro umbral.

*Seccion generada por `experiments/v51_00_utilidad_y_asimetria.py`,
`experiments/v51_01_selector_utilidad.py`,
`experiments/v51_02_accion_utilidad_esperada.py` y
`experiments/v51_03_integrar_notebook.py`. Sin push: la rama sigue local.*


---

## Addendum P52 (2026-09-28) - El techo de la politica observable es negativo

### La pregunta de fondo, respondida

> ?Existe una politica de decision basada unicamente en senales observables que tenga un
> techo materialmente superior a V03?

**No. Y se puede demostrar con una cota, no con un modelo que salio mal.**

### La cadena de techos

| objeto | corrected_cv | delta vs V03 | usa `y_true` |
|---|---:|---:|---|
| V03 (inaccion) | 0,7772114525 | +0,0000000000 | no |
| estrato, accion del training fold | 0,7549322355 | -0,0222792170 | solo en entrenamiento |
| **STRATUM_ORACLE** | **0,7753804469** | **-0,0018310056** | si |
| TRUE_ORACLE | 0,9870178614 | +0,2098064089 | si |

El `STRATUM_ORACLE` es la mejor politica **constante dentro de estratos observables**,
con la accion de cada estrato elegida mirando `y_true`. Usa 18 estratos definidos solo con
la prediccion de V03, el bin de margen y el numero de canales que discrepan.

**Es una cota, no un experimento.** Si esa politica pierde frente a no hacer nada, ninguna
politica condicional sobre variables observables puede ganar.

### Donde esta el +0,2098, y por que es inalcanzable

| tramo | magnitud | que mide |
|---|---:|---|
| TRUE_ORACLE -> STRATUM_ORACLE | **+0,211637** | coste de obligar a la politica a ser constante en un estrato |
| STRATUM_ORACLE -> V03 | **-0,001831** | lo que una politica estratificada puede ganar |

Practicamente todo el oracle esta en la **granularidad de celda**. Exige una accion
distinta en cada celda, y los estratos observables no dicen cual.

### Rescate exclusivo: la hipotesis de complementariedad, medida

| magnitud | valor |
|---|---:|
| celdas con rescate unico | **78 de 715** |
| HHI del reparto de los 715 rescates | **0,0020** |
| canales con `net_exclusivo` positivo | **2 de 20** |
| pares con exclusividad bilateral | **0 de 190** |
| pares donde los dos canales se contradicen | **0 de 190** |

El HHI de 0,0020 esta **por debajo** del minimo teorico de 1/20 = 0,05, porque los canales
se solapan muchisimo. Y el dato duro: **donde dos canales difieren de V03, nunca difieren
entre si**. No hay nada que repartir, asi que ninguna combinacion puede aportar de forma
exclusiva. La hipotesis era razonable y queda cerrada con evidencia.

### Correcciones a P51

1. La elevacion necesaria al bajar de 1 a 0 era **1,62x**, no 1,03x. P51 dividio
   `(1 - 0,6108)` por la tasa base, cuando el rescate al bajar es `P(y=0)` y su equilibrio
   es el propio `0,6108`. La direccion 1->0 era **mas dificil** de lo reportado.
2. P51 aplico el umbral **0,5** donde la teoria da **0,3892** al subir y **0,6108** al bajar,
   luego sobre-cambio justo en la direccion que mas dano hace.
3. P51 mezclaba las dos direcciones en un solo clasificador por canal.

### Una lectura que se descarta a si misma

La curva de valor de la politica sale con net +225 y **no es valor de politica**: la accion
de cada estrato se elige con labels del training fold, de modo que todo cambio es rescate
por construccion. Se reporta marcada como contaminada y la conclusion se apoya unicamente
en el `STRATUM_ORACLE`.

### Decision

`CASE C`. `UTILITY_POLICY = CLOSED` y `CONTEXTUAL_SELECTION = CLOSED`.

El motivo final ya no es "no lo probamos". Es: **la mejor politica estratificada posible,
con permiso de mirar `y_true`, pierde**. Por tanto la linea contextual esta cerrada por
una cota negativa, y cualquier intento futuro de seleccion contextual sobre este pool
tendria que demostrar primero que supera 0,7754.

*Seccion generada por `experiments/v52_00_auditoria_y_techo.py` y
`experiments/v52_01_integrar_notebook.py`. Sin push: la rama sigue local.*


---

## Addendum P53 (2026-09-28) - El espacio de acciones se colapsa a 2

### El colapso, que explica casi todo lo anterior

En las **3896 celdas con discrepancia**, todos los canales que discrepan proponen
exactamente el valor `1 - v03_pred`.

| magnitud | valor |
|---|---:|
| celdas con discrepancia | 3896 |
| celdas con valor propuesto **distinto** de `1 - v03_pred` | **0** |
| celdas donde dos canales discrepantes se contradicen | **0** |
| **espacio de acciones real** | **2** |

Cambiar al canal 3 o al canal 17 produce **la misma prediccion**. La mascara no puede decir
el valor propuesto, porque el valor no depende de quien discrepa; solo las **probabilidades**
dicen cuan fuerte empuja cada canal. El `TRUE_ORACLE` reconstruido con el colapso da
**0,9870178614**, identico a P50.

### La cadena de techos

| politica | corrected_cv | delta vs V03 |
|---|---:|---:|
| **P0: no hacer nada (V03)** | **0,7772114525** | **+0,0000000000** |
| P1: margen + conteo de discrepancias | 0,7623696498 | -0,0148418027 |
| P2: + mascara exacta | 0,7566874656 | -0,0205239869 |
| P3: + identidad de etiqueta | 0,7588669459 | -0,0183445066 |
| P4: + probabilidades de los canales | 0,7712584603 | -0,0059529922 |
| P52 STRATUM_ORACLE (18 estratos) | 0,7753804469 | -0,0018310056 |
| TRUE_ORACLE | 0,9870178614 | +0,2098064089 |

**El mejor techo observable es P0.** Toda particion informativa empeora a V03. Y el
diagnostico de soporte lo explica: con soporte minimo 20 en training, los cambios caen a
**cero**. Los patrones con expectativa positiva en retrospectiva son los pequenos, luego no
generalizan.

### El canje, medido donde existe

Sobre las 3896 celdas con discrepancia:

| tipo | n | rescate | dano | net |
|---|---:|---:|---:|---:|
| un solo canal | 959 | 8,13 pct | 91,87 pct | **-803** |
| dos o mas | 2937 | 21,69 pct | 78,31 pct | **-1663** |

El dano domina por cuatro a uno. Voltear a ciegas pierde siempre.

### Tres correcciones a P52

1. **El `STRATUM_ORACLE` no estaba contaminado; la pregunta estaba mal planteada.** Elegir
   la accion con `y_true` es justamente lo que hace que sea un **techo holgado por
   construccion**. Si un techo holgado cae por debajo de V03, todo lo mas cenido tambien
   cae. La conclusion de P52 se sostenia.
2. **La exclusividad bilateral estaba mal calculada.** P52 reporto `0 pares`; P53 obtiene
   **189 de 190**. El cero era un artefacto de restringir la medicion al conjunto donde ambos
   canales discrepan, que por el colapso es degenerado.
3. **El HHI estaba mal definido.** `sum((exclusivo_i / 715)^2)` usa proporciones que **no
   suman 1**. P53 reporta las dos metricas por separado:

| metrica | valor |
|---|---:|
| HHI estandar de los 78 exclusivos | **0,360289** (minimo teorico 0,090909) |
| cuota de exclusivos sobre los 715 | **10,91 pct** |

### Distribucion de mascaras

1126 patrones distintos sobre 3896 celdas, frecuencia mediana **1**, maximo 658, **843
patrones singleton**. La mascara esta muy fragmentada, y esa fragmentacion es la causa
directa de que las politicas fine-grained no generalicen.

### Interaccionalidad: existe, y no sirve

Diez canales tienen soporte suficiente en ambos regimenes, con diferencias de tasa de
rescate entre **-0,2584** y **+0,1069** segun actuen solos o con otros. Hay estructura del
tipo "canal A util cuando actua solo, canal B util cuando A tambien discrepa". Pero no llega
a producir una politica positiva.

### Decision

`CASE C`. `MASK_BASED_SELECTION = CLOSED`. **No se entrena modelo**, porque el techo no supera
V03 + 0,003.

**Alcance de la afirmacion.** Lo que la evidencia sostiene: ninguna politica basada en la
mascara de desacuerdo, el margen, el numero de discrepancias, la identidad de la etiqueta y
las probabilidades de los canales supera V03 en evaluacion cross-fitted. Y como el colapso
demuestra que el espacio de acciones real tiene **dos** elementos, esto es bastante mas
fuerte que lo que P52 podia afirmar. Lo que **no** se afirma, y no debe afirmarse, es que
ninguna politica imaginable supere V03.

*Seccion generada por `experiments/v53_00_politica_fina.py` y
`experiments/v53_01_integrar_notebook.py`. Sin push: la rama sigue local.*


---

## Addendum P54 (2026-09-28) - El residuo es frontera de etiqueta, no informacion ausente

### La pregunta, respondida

> ?Los errores restantes de V03 son errores de representacion, errores de decision, o casos
> donde el texto disponible simplemente no contiene suficiente informacion?

**Ninguna de las tres, en su forma habitual. Son, sobre todo, desacuerdos de frontera de
etiqueta.**

### El soporte, antes que nada

| magnitud | valor |
|---|---:|
| celdas de C | 42 |
| **tweets distintos** | **33** |
| celdas de C que vienen de 3 tweets con las tres etiquetas mal | 9 |

El soporte efectivo es **33 observaciones, no 42**. Toda enrichment de este residuo debe
leerse con esa cifra delante.

### Composicion

FP (V03=1, y=0) **22** ? FN (V03=0, y=1) **20**. Y el reparto por etiqueta es desigual:

| etiqueta | errores | no rescatables | pct |
|---|---:|---:|---:|
| HS | 316 | 9 | **2,85 pct** |
| TR | 164 | 12 | 7,32 pct |
| AG | 277 | 21 | 7,58 pct |

El **85,7 por ciento** de C cae en margen alto, y **36 de 42** son errores de alta
confianza. P49 lo sugirio, P54 lo confirma.

### Una impresion mia que los datos corrigen

Leer las 42 celdas sugiere con fuerza que el residuo es `insulto lexico sin escalada`. Al
medirlo:

| | pct con lexico de insulto |
|---|---:|
| C (no rescatable) | 66,67 pct |
| A (correcto) | 53,36 pct |
| **enrichment C/A** | **1,25x**, p = 0,0566 |

**No discrimina.** El insulto esta en la mitad de todas las celdas, correctas e
incorrectas por igual. La lectura cualitativa era correcta sobre el subconjunto de AG y
equivocada sobre el conjunto.

### El mayor candidato, y por que no es hipotesis

La **negacion** es el atributo con mas fuerza en todo el residuo:

| atributo | n en C | pct C | pct A | enrichment C/A | p |
|---|---:|---:|---:|---:|---:|
| negacion | 17 | 40,48 pct | 22,35 pct | **1,81** | **0,0064** |
| marcador de contraste | 5 | 11,90 pct | 5,63 pct | 2,11 | 0,0858 |
| negacion **y** contraste | **1** | 2,38 pct | 3,11 pct | **0,76** | 0,7353 |

La lectura cualitativa proponia un mecanismo *concedente*: la negacion licencia lo que
viene despues, como en ?no soy racista pero si persona?. **Queda refutado**: una sola celda
de 42 lo cumple, con enrichment 0,76x. Y la negacion tambien tiene enrichment de **rescate**
sobre B de 1,55x, luego **marca celdas dificiles en general, no un modo de fallo concreto**.
El mecanismo se declara **sin explicar**.

### Los otros dos candidatos, y el argumento del valor absoluto

| candidato | n en C | enrichment C/A | p | valor en macro-F1 |
|---|---:|---:|---:|---:|
| `mi + insulto` (posesivo) | **4** | 7,33 | 0,0022 | ~0,0005 |
| grupo marginal (`sin papeles`, `inmigrante`) | **9** | 2,68 | 0,0052 | ~0,0011 |

El lexico de grupo marginal **no aparece en ningun script del proyecto**: `sin_papeles` e
`indocumentad` dan 0 ficheros. El mecanismo no se ha probado. Pero **26,75 celdas valen
~0,003 de macro-F1**, luego ninguno llega al margen, y por el resultado de P51 a P53 una
regla que acierte esas celdas tambien dana otras.

### Taxonomia retrospectiva de las 42 celdas

| categoria | n | pct de C |
|---|---:|---:|
| FP: insulto lexico sin escalada | 21 | 50,0 pct |
| FN: narrativa sobre grupo marginal | 9 | 21,4 pct |
| FN: insulto con agresividad no activada | 6 | 14,3 pct |
| FN: agresividad o target no lexicalizada | 4 | 9,5 pct |
| FP: error no lexicalizado | 1 | 2,4 pct |
| FN: discurso citado o atribuido | 1 | 2,4 pct |

Marcada como **retrospectiva**: el modelo vio texto y etiqueta a la vez, luego describe y
no infiere.

### Decision

`P54_RESEARCH_GAP = NONE`, con **0 hipotesis nuevas**. La conclusion es especifica:

1. **No es falta de informacion en el texto.** El 66,67 por ciento de C ya contiene el
   lexico de insulto, y el reparto por etiqueta (2,85 / 7,32 / 7,58 por ciento) es desigual
   de un modo que el texto si explica.
2. **Es, en su mayor parte, desacuerdo de frontera de etiqueta.** Insulto lexico con
   etiqueta 0, y narrativa sobre grupo con etiqueta 1. En ambos casos el texto es
   inequivoco y la lectura del anotador es una convencion que no se deriva del texto.
3. **No queda fenomeno semantico con soporte suficiente y valor material.**

**Lo que no se afirma.** No se afirma que sea imposible mejorar. Se afirma que en este
residuo no hay mecanismo semantico con soporte y valor material, y que la via de datos
observables quedo cerrada por P52 y P53 con una **cota**, no por falta de intento.

*Seccion generada por `experiments/v54_00_atlas_no_rescatable.py` y
`experiments/v54_01_integrar_notebook.py`. Sin push: la rama sigue local.*


---

## Addendum P55 (2026-09-28) - La referencia de V03 mezcla dos protocolos

### El pipeline real

```text
raw train 1731 -> dev 1483 / holdout 248

A) 04_refit_submissions_v02_v03.py
     grupos = near-duplicado (D-004)
     vectorizador y modelo DENTRO de cada pliegue
     -> oof_mod03b.npz
     -> umbrales 0,44 / 0,51 / 0,46 sobre ESE OOF
     -> refit sobre el 100 pct del train con ESOS umbrales
     -> submission V03

B) 14_oof_corregido_v2.py
     grupos = near-duplicate UNION pseudo-autor (D-016)
     -> v2_oof_base_corrected.npz
     -> corrected_cv = 0,7772114525 con LOS UMBRALES DE A)
```

**La discrepancia:** los dos numeros de la referencia de V03 vienen de protocolos de
agrupacion distintos, y D-004 es el mas debil porque deja que un pseudo-autor cruce
pliegues. Cuando D-016 corrigio el protocolo, el `corrected_cv` se recalculo conservando los
umbrales viejos.

### El sesgo es sistematico

| etiqueta | umbrales anidados por pliegue | mediana | V03 | dentro del rango |
|---|---|---:|---:|---|
| HS | 0,40 0,41 0,40 0,41 0,41 | 0,41 | **0,44** | **NO, en los 5** |
| TR | 0,50 0,48 0,50 0,48 0,48 | 0,48 | **0,51** | **NO, en los 5** |
| AG | 0,47 0,45 0,45 0,45 0,45 | 0,45 | 0,46 | si |

### La magnitud honesta esta bajo el margen

| metodo | macro-F1 | delta vs V03 |
|---|---:|---:|
| umbrales de V03 | 0,7772114525 | +0,0000000000 |
| optimos **in-sample** | 0,7808400798 | **+0,003629** |
| **anidados (honesto)** | **0,7781367700** | **+0,000925** |

El numero in-sample habria superado el margen de adopcion. **Es una trampa**: el umbral se
eligio sobre el mismo OOF que se puntua. El anidado, con el umbral elegido en cuatro
pliegues y aplicado al quinto, queda en +0,000925, **por debajo de 0,003**.

### La consecuencia operativa

| umbrales | HS | TR | AG |
|---|---:|---:|---:|
| de V03 | 0,4572 | 0,3001 | 0,3776 |
| anidados | 0,4970 | 0,3075 | 0,3810 |

Los umbrales de V03 son demasiado altos: el OOF produce unos 4 puntos por debajo de
positivos en HS que los umbrales del propio protocolo piden. Es **consistente en
direccion** con que el LB saliera 0,0058 por encima del OOF, aunque **no lo demuestra**.

### Lo que la auditoria no encontro

- **Vectorizador dentro del pliegue**: verificado en el codigo
  (`vectorizador.fit_transform(textos_entrenamiento)`), luego **no** hay leakage de
  representacion.
- **Prevalencias homogeneas** entre pliegues, luego ninguno domina.
- **El mecanismo no es el tamano de entrenamiento**: con 80 %, 90 % y 100 % de los grupos del
  training fold, los umbrales son practicamente identicos.
- **La brecha OOF a LB** es de 0,25 sigma. Se sabe el signo, se desconoce la causa, y ninguna
  consulta a Kaggle es legal en esta fase.

### Integridad de artefactos

`features/oof_mod03b.npz`, que produjo los umbrales adoptados, **no existe en disco**. Los
umbrales de V03 **no son auditables**. El sha256 de la submission coincide y el
`corrected_cv` se reproduce exactamente.

### Decision

`P55_PROCEDURAL_GAP = B`, clasificado como **`BUG_FOUND`** y **no** como mejora, con
original y corregido separados:

```text
ORIGINAL_RESULT   corrected_cv = 0,7772114525   umbrales 0,44/0,51/0,46 de D-004
CORRECTED_RESULT  corrected_cv = 0,7781367700   umbrales anidados de D-016
delta             +0,000925                     por debajo del margen de 0,003
```

**0 experimentos nuevos, 0 submissions.** La correccion es legal y el mecanismo es concreto,
pero su magnitud medida no alcanza el margen, y V03 permanece congelado e intacto.

### Recomendacion estrategica final

> V03 permanece como mejor resultado verificado y no se identificaron defectos
> procedimentales ni mecanismos de mejora pendientes dentro del pipeline permitido.

Con dos matices que la evidencia si permite y que conviene no perder:

1. **Se identifico un defecto procedimental** cuantificado (+0,000925) que esta por debajo del
   margen de adopcion. No se corrigio porque corregirlo no mejora el resultado de forma
   adoptable, y V03 esta congelado.
2. **No se afirma que V03 sea globalmente optimo.** La brecha de 0,25 sigma con el
   leaderboard queda sin explicar, y el techo de 0,9870 que P50 midio sigue sin alcanzarse.
   Cerrar la linea procedimental no es cerrar el techo.

*Seccion generada por `experiments/v55_00_auditoria_pipeline.py` y
`experiments/v55_01_integrar_notebook.py`. Sin push: la rama sigue local.*


---

## Addendum P56 (2026-09-28) - El sesgo de umbrales no se sostiene

### Reproduccion de P55

`P55_CORRECTED_RESULT_REPRODUCED = TRUE`, 0,7781367700 exacto. La correccion procedimental
de P55 es un hecho historico real: los umbrales se adoptaron bajo el protocolo D-004, mas
debil que el D-016 con el que se reporta la metrica.

### El anidamiento completo corrige el diagnostico

P55 uso un procedimiento que **no** era anidamiento completo: tomaba el umbral de las
probabilidades out-of-fold de cuatro pliegues, ya producidas por modelos entrenados. P56
re-ajusta vectorizador y modelo dentro de un split interior agrupado, y el resultado cambia:

| etiqueta | estimaciones interiores | mediana | std | min | max | V03 | ?V03 en el rango? |
|---|---|---:|---:|---:|---:|---:|---|
| HS | 0,40 0,41 0,42 0,41 0,45 | 0,41 | 0,0192 | 0,40 | 0,45 | 0,44 | **si** |
| TR | 0,47 0,46 0,46 0,51 0,46 | 0,46 | 0,0217 | 0,46 | 0,51 | 0,51 | **si** |
| AG | 0,40 0,48 0,43 0,51 0,45 | 0,45 | 0,0428 | 0,40 | 0,51 | 0,46 | **si** |

**Los umbrales de V03 caen dentro del rango en las tres etiquetas.** El sesgo sistematico de
+0,03 ?fuera del rango en los cinco pliegues? era un artefacto del procedimiento mas simple,
que producia rangos artificialmente estrechos.

### La cifra honesta es negativa

| politica | macro-F1 | delta vs V03 |
|---|---:|---:|
| T0 historico (0,44/0,51/0,46) | 0,7772114525 | +0,0000000000 |
| T1 mediana aplicada **globalmente** | 0,7798514633 | +0,0026400107 |
| **umbral por pliegue exterior (honesto)** | **0,7745478730** | **-0,0026635795** |
| P55 simple, referencia | 0,7781367700 | +0,0009253174 |

**El optimismo de definir el umbral con datos de los cinco pliegues y evaluarlo en los cinco
es de +0,0053: el doble de la mejora aparente.** Por eso la cifra que decide es la del
umbral de cada pliegue exterior aplicado a ese mismo pliegue.

### Tres senales independientes

| senal | valor |
|---|---|
| net a nivel de celda, T1 contra T0 | **-11** (43 rescates, 54 danos) |
| net por etiqueta | HS -5, TR -6, AG 0 |
| bootstrap del delta de T1 | mediana +0,002553, IC95 [-0,003456, +0,008989] |
| el IC95 incluye 0 | **si** |
| pliegues con delta positivo | 4 de 5, con el pliegue 3 en **-0,0310** |

El F1 sube mientras el recuento de celdas baja. No es contradictorio: macro-F1 promedia
tres etiquetas y un cambio de umbral reordena las positivas. Pero el recuento de celdas es
la medida honesta del efecto, y es negativo en las tres etiquetas.

### Tasa positiva

| etiqueta | T0 | T1 | diferencia |
|---|---:|---:|---:|
| HS | 0,4572 | 0,4943 | +0,0371 |
| TR | 0,3001 | 0,3216 | +0,0216 |
| AG | 0,3776 | 0,3844 | +0,0067 |

T1 produce muchas mas positivas y aun asi el F1 no mejora de forma robusta.

### Decision

`CASE B`. `THRESHOLD_TRANSFER_BIAS = NOT_CONFIRMED`. `STATUS = CLOSED`. **No se crea
candidato local y no se genera submission.**

### Que se corrige de P55

| afirmacion de P55 | estado tras P56 |
|---|---|
| los umbrales se adoptaron bajo D-004 | **se mantiene**, es un hecho |
| estan fuera del rango en los cinco pliegues | **CORREGIDO**: caen dentro |
| el sesgo es sistematico, +0,03 en HS y TR | **CORREGIDO**: no hay sesgo sistematico |
| la correccion da +0,000925 | **CORREGIDO**: la cifra honesta es -0,002664 |
| `BUG_FOUND` con impacto bajo el margen | **se mantiene como hecho**, el impacto es nulo o negativo |

El defecto procedimental de P55 **sigue siendo real como hecho historico**. Lo que se
corrige es su magnitud, no su existencia. Y la conclusion estrategica queda mas fuerte que
en P55: la linea procedimental esta cerrada por medicion, no por abandono.

*Seccion generada por `experiments/v56_00_transferencia_umbrales.py` y
`experiments/v56_01_integrar_notebook.py`. Sin push: la rama sigue local.*


---

## Addendum P57 (2026-09-28) - Cierre estrategico y plan final

> Fase de auditoria y sintesis. **0 modelos, 0 entrenamientos, 0 submissions, 0 consultas a
> Kaggle, 0 uso de holdout, 0 uso de test.** Decision **D-128**. No se registro ninguna fila
> nueva en `experiments.csv`.

### Baseline congelado

```text
id           v03_lr_union_thr
repr         TF-IDF word(1,2) + char_wb(3,5), min_df=2, sublinear_tf
clasificador LogisticRegression C=2.0, class_weight=balanced, liblinear, max_iter=1000, seed 42
umbrales     HS 0.44 | TR 0.51 | AG 0.46
corrected_cv 0.7772114525165721
LB           0.78306
sha256       7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
```

### 1. La jerarquia de magnitudes, con la incompatibilidad explicita

| magnitud | valor | experimento | comparable con V03 |
|---|---:|---|---|
| best_architectural_CV (macro-6, diagnostica) | 0.8743 | S08 | **NO** |
| best_holdout_CV | 0.8465 | S08 | **NO** |
| best_CV protocolo D-004 | 0.8039 | S05 / v05_optuna | **NO** |
| best comparable D-016 | 0.7904 | svd300_rbf | SI, **con seleccion** |
| best anidado D-016 | 0.7885 | E2-13-META-ENT-RIDGE | SI, sin seleccion |
| **V03 corrected_cv** | **0.7772114525165721** | V03 | SI |
| **best verified LB** | **0.78306** | v03_lr_union_thr | SI |
| **current champion** | **V03** | sha256 `7d7478f2...` | SI |
| remaining submission budget | limite diario exacto desconocido | 0 hoy, reserva propia 3 | n/a |

### 2. El problema de lectura del registro

`experiments.csv` mezcla en `f1` y `macro_f1` **al menos tres metricas y tres protocolos**:

- `macro_f1` contiene **macro-6** en la era V2/E2/MOD/TH/ENS y la metrica del proyecto en la era
  V10+.
- `f1` contiene **holdout de 248 filas** en S08, **D-004** en la era antigua y **D-016** en la
  corregida.

Medido sobre el mismo OOF de V03: metrica del proyecto 0.7772114525, macro-6 0.8182213363,
**diferencia +0.04101**. Y `leaderboard_log.csv` registra para V03 un `local_f1` de 0.7989,
**+0.021689** por encima del `corrected_cv` canonico, que es el coste de medir con D-004.

**Cualquier tabla de "mejor CV" construida tomando el maximo de esas columnas es falsa.** La
clasificacion `metrica` / `protocolo` / `poblacion` / `anidado` se anade en
`reports/p57_final_experiment_inventory.csv` (323 filas). El registro historico no se modifica.

### 3. Submissions: el recuento era erroneo

| magnitud | valor |
|---|---:|
| envios historicos totales | **23** |
| con LB verificado en el log | 22 |
| envio real NO registrado en el log | 1 |
| ficheros en disco | 37 |
| generados y nunca enviados | 14 |
| maximo LB verificado | 0.78306 |
| envios hoy, 2026-09-28 | 0 |

`experiments.csv` V33-03 documenta un envio real y aceptado, **ID 56608788**, 2026-09-27,
`toxic_submission_v32_05_url_length.csv`, LB 0.78306, delta **exactamente 0**. **No esta en el
log.** El total es 23, no 22.

### 4. El patron que gobierna la estrategia

| familia | envios | LB | veredicto |
|---|---:|---|---|
| kernels | 1 | 0.77938 | RECHAZADO |
| label_specialization | 1 | 0.77781 | RECHAZADO |
| ensembles | 3 | 0.7751, 0.7747, 0.77122 | los tres a la baja |
| stacking / meta-modelos | 5 | 0.78163, 0.77895, 0.77978, 0.78150, 0.77663 | los cinco a la baja |
| structural_features | 1 | 0.78306, delta 0 | sin efecto |

**Diez envios de candidatos con mecanismo nuevo. Los diez transfirieron a la baja.** Y el mejor
CV comparable del proyecto **no es el campeon**.

### 5. Ningun artefacto cumple los cinco criterios de candidato

| artefacto | que le falta |
|---|---|
| v23_01 svd300_rbf, 0.7904 | ya rechazado: enviado, LB 0.77938 |
| v08_meta_v2, 0.7874 | ya rechazado: enviado, LB 0.78163 |
| v20_01 meta 36 features, 0.7882 | ya rechazado dos veces: holdout y LB 0.77663 |
| E2-13-META-ENT-RIDGE, 0.7885 anidado | cerrado por sparsity, nunca verificado fuera |
| umbrales anidados de P56 | NOT_CONFIRMED, delta honesto negativo |
| v16_ch2 prior shift EM | transductivo |

### 6. Estado de las 22 familias

20 cerradas por resultado negativo (A), sin transferencia (C) o inexistencia (E). Dos con
matiz:

- **PLS: tipo B.** El experimento se invalido por implementacion, luego su numero no es
  evidencia. El resultado verdadero (V43-02) es -0.0137423671.
- **Estructura de etiquetas en V13-02 y V14-00: nunca ejecutadas**, 0 de 4 experimentos,
  cerradas por condicion de parada. **No cabe decir "no funciono".**

### 7. Conclusiones de mechanism que P56, P47 y P54 ya fijaron

```text
THRESHOLD_TRANSFER   P55 +0.000925 | P56 honesto -0.002664 | CLOSED
CONTEXTUAL_SELECTION techo observable -0.001831, oracel +0.209806 | CLOSED
DATA_EXPANSION       0 filas oficiales sin usar | CLOSED
residual             42 celdas = 0,95 por ciento, en 33 tweets
```

La formulacion sobre los canales: **la senal existe y es enorme**, y **ningun selector probado
la convirtio en mejora neta positiva**. No se afirma que no haya senal.

### 8. Artefactos que no deben volver a usarse

`v17` y `v18` y `v16_ch2` (transductivas), `v12_descartado_meta36_downweight` (rechazada por
holdout), `v20_03_reproduccion_v03` (invalida), `v32_05_url_length` (gastada, delta 0). Ninguna
se modifica. Ninguna se vuelve a enviar.

### 9. Formulacion final obligatoria

> V03 remains the best verified leaderboard result and the frozen architectural baseline. The
> investigated legal mechanisms did not produce a robust transferable improvement, and no
> additional mechanism with sufficient evidence has currently been identified.

```text
best verified result  !=  proof of global optimum
```

Verificado: 0.78306 es el maximo del leaderboard en 23 envios; 0.7772114525165721 es el CV
corregido de la configuracion congelada.

**No afirmado:** que V03 sea globalmente optimo. La brecha OOF-LB de 0.00584855 sigue sin
mecanismo identificado, y el techo del oracel de +0.2098064089 no se ha alcanzado **ni se ha
descartado como alcanzable**. Que ningun candidato con mejor CV local haya transferido es
evidencia **en contra de la optimizacion local** y a favor de V03 como eleccion; **no es
evidencia de que no exista una mejor**.

### 10. Las tres estrategias, sin elegir

| estrategia | a favor | en contra | riesgo | coste |
|---|---|---|---|---|
| **A. cerrar con V03** | 0 de 23 lo superaron; los 10 candidatos a la baja; 22 familias cerradas | techo del oracel +0.2098 sin alcanzar; brecha OOF-LB sin explicar | bajo: el leaderboard conserva la mejor puntuacion | cero |
| **B. candidato historico** | E2-13 con +0.0113 anidado y sin contradiccion externa | los tres con delta comparable tienen veredicto contrario; patron 10/10 | alto | 1 submission, y **no hay candidato que cumpla los cinco criterios** |
| **C. ultima investigacion dirigida** | brecha OOF-LB y asimetria dev-holdout sin caracterizar | ninguna hipotesis cumple las seis condiciones de reapertura; banda de particion 0.017 | medio | alto, no medido |

**La decision competitiva queda pendiente de una instruccion posterior. P57 no la toma y se
detiene. No se abre P58.**

*Seccion generada por `experiments/v57_00_sintesis_estrategica.py` y
`experiments/v57_01_integrar_notebook.py`. Sin push: la rama sigue local.*


---

## Addendum P58 (2026-09-28) - Auditoria final de candidatos existentes

> Auditoria pura. **0 modelos, 0 entrenamientos, 0 submissions, 0 consultas a Kaggle, 0 holdout,
> 0 test.** **0 filas nuevas en `experiments.csv`** (siguen siendo 323). Decision **D-129**.

### Pregunta y respuesta

> ?Existe ya, dentro de los artefactos que hemos producido, algun candidato suficientemente
> defendible como para justificar una unica sonda final?

**No. `FINALIST = NONE`. `CASE A`.** Mantener V03.

### La correccion a P57

P57 escribio que `E2-13-META-ENT-RIDGE` era el unico candidato con delta honesto anidado
positivo y sin veredicto externo. **Es incorrecto**, y la auditoria lo demuestra:

| | valor |
|---|---|
| `v2_meta_oof.npz` declara | `META-BASES-RIDGE`, ridge alpha 3, 44 features |
| eso es | **el modelo enviado como `v08_meta_v2`, LB 0,78163** |
| cobertura del canal de entidades | **0,52 por ciento** frente a un umbral del 20 por ciento |
| de 1079 menciones sobreviven | 6 |
| de 255 hashtags sobrevive | 1 |
| unico dominio que pasa el filtro | `t.co`, 1 unico en desarrollo |
| incremento **sobre su comparador** | **+0,0011, 3 de 5 pliegues**, complementariedad ?0,0003 |
| su delta de +0,0113 | **es de la familia META, no suyo** |

**E2-13 no es un mecanismo nuevo: es `v08_meta_v2` mas un canal que cubre el medio por ciento de
las instancias.** El cierre por sparsity es metodologico y esta bien justificado, porque el
incremento medido confirma la causa.

### Descomposicion celular, recomputada sin reentrenar

```text
candidato                              ccv       delta  plieg  cambian    r    d  neto  acierto  LB
META-BASES-RIDGE (= v08_meta_v2)  0.787357  +0.010146   4/5     246  132  114   +18   0.5366  0.78163
A_cost_sensitive_downweight       0.785416  +0.008205   5/5     173   99   74   +25   0.5723  sin veredicto
B_meta36                          0.782774  +0.005562   4/5     186  115   71   +44   0.6183  sin veredicto
C_meta36_downweight               0.788173  +0.010962   4/5     212  134   78   +56   0.6321  holdout
D_svd300_rbf                      0.790410  +0.013198   5/5     240  136  104   +32   0.5667  0.77938
```

Reproducen `reports/v4_candidates.json` al sexto decimal, lo que prueba que los tensores en
disco son los que alimentaron el gate pre-registrado.

### Los siete criterios

**El que decide es el 6, `artefacto de prediccion completo`.** Solo tres de los cinco tienen
CSV, y los tres ya fueron enviados o rechazados por el holdout.

| candidato | 6 artefacto | 7 no rechazado por LB | FINALISTA |
|---|:-:|:-:|:-:|
| META-BASES-RIDGE | si | **NO** | **NO** |
| A_cost_sensitive_downweight | **NO** | si | **NO** |
| B_meta36 | **NO** | si | **NO** |
| C_meta36_downweight | si | **NO** | **NO** |
| D_svd300_rbf | si | **NO** | **NO** |

### El caso limite: B_meta36

El unico que pasa el gate D-016 pre-registrado (IC95 [+0,003147, +0,016633], 4 de 5 pliegues,
acierto 0,6183, el mas alto de los cinco) y **no tiene ningun veredicto externo**. No es
finalista por tres razones independientes:

1. **No existe artefacto de prediccion**, y la directiva 12 prohibe crear uno.
2. **El holdout esta agotado 5 de 5**, luego no puede pasar por el filtro que el proyecto uso
   para decidir en la fase final.
3. **Su hermano `C` es estadisticamente el mismo modelo que `v08_meta_v2`**
   (`delta_vs_meta_v08 = ?4,25e-05`), que dio LB 0,78163.

**Pasa cinco de siete y falla en el sexto por ausencia de artefacto, no por defecto del modelo.**
Es el `quasi-finalista` del proyecto.

### Panorama de los 31 candidatos con corrected_cv superior a V03

| categoria | n |
|---|---:|
| con LB verificado y **positivo** | **0** |
| con LB verificado y negativo | 5 |
| rechazados por holdout | 1 |
| **sin veredicto externo** | **2** (A y B, ninguno con artefacto) |
| familias de embeddings cerradas por inercia | 10 |
| por debajo de su gate | 3 |
| mediciones, no modelos | 1 |

### D_svd300_rbf: `CLOSED`

El mejor `corrected_cv` del proyecto, 5 de 5 pliegues, 136 rescates contra 104 danos y McNemar
p=0,0452, **perdio 0,00368 en el externo**. La firma mecanica esta en el soporte: 60 a 76 por
ciento de vectores de soporte, la marca de un kernel RBF con `gamma=scale` sobre 300 dimensiones
que memoriza. El LB ya aporto la evidencia negativa suficiente.

### Integridad de artefactos

Los **37 de 37** CSV de `submissions/` pasan la validacion formal: 1154 filas, columnas
`id,HS,TR,AG`, ids y orden identicos a `toxic_test.csv`, binarios, sin nulos, sha256.
`v20_03_reproduccion_v03.csv` es **bit a bit identico** a V03. **Ningun artefacto se modifica.**

### Presupuesto

```text
local_reservation   3
competition_limit   UNKNOWN      no se consulta Kaggle
envios historicos   23
envios hoy          0
```

### Formulacion final

> V03 remains the best verified leaderboard result.

```text
best verified result  !=  proof of global optimum
```

**Lo que la auditoria SOSTIENE:** que el conjunto de artefactos que hemos producido **no
contiene** un candidato que justifique una sonda final.

**Lo que NO SOSTIENE:** que no exista una mejora defendible. Son afirmaciones sobre estados
distintos, y confundirlas seria el error de razonamiento mas caro disponible ahora mismo.

*Seccion generada por `experiments/v58_00_auditoria_candidatos.py` y
`experiments/v58_01_integrar_notebook.py`. Sin push: la rama sigue local.*


---

## Addendum P59 - Frontier research (2026-09-28)

> 0 submissions en toda la fase, 0 consultas a Kaggle, 0 holdout, 0 test labels.
> V03 congelado: `corrected_cv 0.7772114525165721`, LB 0.78306.
> Decisiones **D-130** (track A). `experiments.csv`: 323 -> 324.

### El gate de novedad hizo su trabajo antes de gastar un solo experimento

> **P59-G no se ejecuta.** V8-03 ya implemento lo que el parrafo 33 propone: seleccion
> por peso alto bajo submuestras, a tres umbrales, con IC95 pareado. Negativo y
> monotono (?0,003308 / ?0,004628 / ?0,018890; 1/5, 1/5, 0/5) y cerrado de forma
> explicita sin reintento.

El registro **verifica** sus precedentes contra `experiments.csv` y aborta si la
cifra citada no esta en la fila real. Eso cazo dos transcripciones erroneas
propias. Un precedente citado de memoria es evidencia inventada.

### P59-A Voronoi ? `CLOSED`, 1 experimento

| resultado | valor | IC95 |
|---|---:|---|
| AUC del score contra error de V03 | **0,725194** | [0,7061, 0,7446], excluye 0,5 |
| AUC del score contra celda no rescatable | 0,471798 | [0,3954, 0,5469], **incluye 0,5** |

```text
A  V03 crudo                     corrected_cv 0,7772114525
B  LR([p_v03])                   0,7743581357   -0,0028533168
C  LR([p_v03, 8 rasgos])         0,7796406441   +0,0024291916
C menos B                        +0,0052825084   con 5 de 5 pliegues
IC95 pareado de C-A              [-0,008991, +0,000674]   incluye cero
acierto en las celdas que cambian 0,4297   (por debajo de 0,5)
rescate 55 | damage 73 | neto -18
de las 42 celdas no rescatables, rescata 1
```

H1 (delta ? +0,003) y H2 (4/5 pliegues) fallan. El mecanismo del fallo es que el
acierto en las celdas modificadas queda **por debajo del azar**: delta de metrica
positivo y neto negativo a la vez. El filtro de detectabilidad de V8-02, que
exigia ~0,55, habria bloqueado el envio igualmente.

**El matiz que no conviene perder:** C supera a B en **+0,005283 con 5 de 5
pliegues**. La geometria aporta algo real y medible. Lo que no hace es mejorar el
punto de operacion de V03, cuya puntuacion cruda ya es mejor que cualquier reajuste.

### Formulacion

> V03 remains the best verified leaderboard result.

```text
SOSTIENE   la geometria local explica que celdas fallan, y no se explotaba
NO SOSTIENE que exista una mejora de decision disponible por esa via
```

La segunda linea es la que manda. Predecir la dificultad de un ejemplo no es lo
mismo que mejorar la decision, y V6-07 ya lo habia medido con otras senales.

*Seccion generada por `experiments/p59_0*.py` y `p59_90_integrar_notebook.py`. Sin push.*
<!-- P60 2026-09-28 -->
<!-- P60 2026-09-28 -->
## P60 - cierre de la fase

P60 CLOSED. Doce pistas de literatura multi-etiqueta, 8 experimentos consumidos, 4 pistas cerradas sin experimento, 0 candidatos, 1 marginal, 0 bootstraps, 0 submissions, 0 Kaggle, 0 holdout, 0 test. V03 sigue siendo el resultado verificado. sha256 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e. corrected_cv 0.7772114525165721. LB 0.78306. Detalle en reports/P60_FINAL_RESEARCH_REPORT.md y reports/TOXIC_COMPLETE_RESEARCH_REPORT.md.

<!-- P61 2026-09-28 -->
## P61 (2026-09-28) - Capacidad, diversidad y distribucion de probabilidad

Tres experimentos: `P61-01` LightGBM con Optuna anidado, `P61-02` ensemble con pertenencia
y pesos anidados, `P61-03` segunda particion con control negativo. Ademas la
reconciliacion de los nueve problemas de instrumentacion de P60, la EDA de las
probabilidades OOF de V03 y el inventario con diversidad de los veinte canales del pool de
P50.

```text
P61-01  LightGBM + Optuna anidado    0.7513651502
         familia -0.0302397191
         busqueda +0.0043934168
         total -0.0258463023   2/5   CLOSED
P61-02  ensemble anidado             0.7835159596  +0.0063045070  gate superado
         control de atribucion: el modelo nuevo SUBTRAIA -0,000850
P61-03  segunda particion            test 1 -0.0011527969
         test 2 holdback medio -0.0015908404 con 7/10
         test 3 control negativo +0.0059889238   REJECTED
```

**El hallazgo que mas importa.** El control negativo NO colapsa. Con los tres peores
canales del pool, entre 0,04 y 0,24 por debajo de V03, la ganancia del ensemble sigue
siendo +0.0059889238. Un promediado ponderado que contiene a V03 mejora el OOF casi con
seguridad porque los pesos se eligen para exploitear su residuo sobre las mismas filas, y
el bootstrap no lo detecta porque mide ruido de muestreo y no sesgo de seleccion. Esto
explica de una sola causa el fracaso historico de ENS-01 a ENS-04.

**El diagnostico que queda.** Entre el 60 y el 68 por ciento de los errores de V03 son
errores de alta confianza, con la distancia al umbral POR ENCIMA de la de los aciertos.
El limite de V03 no es de decision ni de calibracion: es de representacion.

```text
no evidence of exploitable remaining model capacity/diversity
V03 remains the best verified leaderboard result.
```
<!-- /P61 2026-09-28 -->

<!-- P62 2026-09-29 -->
## P62 (2026-09-29) - Representacion, estructura linguistica y error de alta confianza

Cinco experimentos, tres pistas cerradas sin gastar nada, y una correccion de lectura.

```text
P62-01  L palabra__POS     -0,009572  1/5  CLOSED
P62-02  M triple lexico   -0,011236  0/5  CLOSED
P62-03  curva de presupuesto   ningun K mejora
P62-04  POS contra lemas   ambos pierden 8/8, el POS ayuda 6/8
P62-05  cobertura min_df=1  -0.0033508936  CLOSED
```

**La interaccion existe y es medible.** La synergy palabra-POS es +0.0657, el
17.41 por ciento de la senal de la palabra. El problema no
es que falte informacion, es que no cabe: repaga el
17.4 por ciento del coste de anadir el canal.

**La cobertura tampoco es la via.** `min_df=2` conserva solo el
28.3 por ciento de las formas superficiales
presentes, y `min_df=1` lo sube al 99.5 por
ciento y resta.

**La correccion.** Yo lei mal mi propia EDA: el perfil de longitud se llamaba
`estructura_corta` pero su cuartil superior es lo largo, luego la delta negativa decia que
V03 es MEJOR en los largos, no peor en los cortos. La comprobacion independiente lo
desmiente: V03 alcanza 0.9639 de AUC en los cortos y su F1
es 0,94 contra 0,72 en los largos. El limite esta en el ORDEN de los textos largos, donde
V28-01 ya habia localizado el techo estructural.

```text
no evidence of exploitable remaining model capacity in the linguistic representation
V03 remains the best verified leaderboard result.
```
<!-- /P62 2026-09-29 -->

<!-- P63 2026-09-29 -->
## P63 (2026-09-29) - Representacion a nivel de registro, estructura del dataset y composicion

Seis experimentos, dos auditorias y un dataset auxiliar. Ninguno supera a V03, y lo
interesante no son los negativos sino que cada uno lleva un mecanismo MEDIDO.

```text
P63-00  ID forensics      D. STRUCTURED_IDENTIFIER, confianza alta, 0 experimentos
P63-02  id predictivo     0,419136 contra 0,424038 de un control de igual dimension
P63-03  relacional        el bloque esta 92,92 por ciento vacio
P63-04  composicion       brecha especifica +0,0083528268, y +0,1055 en solitario
P63-05  texto largo       la brecha en MUY_LARGA es -0,009437
P63-06  integracion       brecha +0,0085891400, MARGINAL
P63-07  residuo           los bloques restan al margen, luego son redundantes con el
                         margen y no con los n-gramas
```

**El id no es un contador.** V15-00 lo afirmo sin medirlo. Train y test son dos muestras
disjuntas del mismo bloque contiguo, con interseccion 0 y union
2885 sobre 4999 posiciones. Es un indice de
coleccion, y su contenido es la posicion en un ensamblado, que no correlaciona con la
etiqueta.

**La composicion en ratios porta senal y no cabe.** Es un resultado distinto del de P62, no
uno contrario: un canal disperso de varios cientos de columnas compite por el presupuesto
L2 de un espacio de 21000 dimensiones, y 42 proporciones acotadas no compiten. Y aun asi
no mejora a V03, porque lo que aportan ya esta implicito en la confianza de V03.

**La hipotesis mas antigua del repositorio queda refutada.** V30-01 escribio que el efecto de
la URL y la mencion cambia de signo con la longitud, y nunca gasto un experimento. El bloque
de posiciones es PEOR que un bloque aleatorio del mismo ancho precisamente en la region
muy larga, con -0.009437. Refutada en su prediccion, no solo
no confirmada.

**El limite que queda es de orden, no de decision.** El margen hasta el techo moviendo solo
el corte va de +0.0000 en la region corta a +0.0890 en la
muy larga, y en la region CORTA es exactamente +0.0000: el umbral de V03 ya
es el optimo ahi sobre las tres etiquetas.

**Dos correcciones cambiaron la direccion de una conclusion** y tienen fila propia en el
registro. El confound de escala de P63-04 paso el delta de -0,1899987838 a
-0,0102687019 y cambio el signo de la brecha. El techo de P63-06 paso el margen de MUY_LARGA
de +0,1589 a +0,0890.

```text
no evidence of exploitable remaining model capacity in the record-level and
aggregate-composition dimensions of this representation

V03 remains the best verified leaderboard result.
```
<!-- /P63 2026-09-29 -->

<!-- P64 2026-09-29 -->
## P64 (2026-09-29) - Representacion del tweet largo y distribucion de la evidencia interna

Cuatro experimentos, dos auditorias, trece figuras. Ninguno supera a V03, y la respuesta
tiene dos mitades que no se pueden separar.

**La estructura interna es REAL.** P59-07 habia explicado su fracaso por cobertura, no por
concepto: con ventana de 32 tokens solo el
9.6 por ciento de las filas tiene mas
de una unidad, mientras que la segmentacion por puntuacion llega al
50.6 por ciento. Un factor de
5.29 en las celdas alcanzables, que no es una variacion
de parametro.

**Y el leave-one-out, que P59 nunca computo, muestra el mecanismo.** El 80,88 por ciento de
los documentos multi segmento de la region MUY_LARGA tienen segmentos que empujan la decision
en signos opuestos, la entropia interna sube de 0,1048 a 0,4565 con la longitud, y la
concentracion cae de 0,7554 a 0,6170. En los tweets largos la decision no descansa en un
segmento dominante sino en un reparto contradictorio. Eso es exactamente lo que una bolsa de
n-gramas aplana.

**Y no aporta.** El bloque de evidencia queda -0.0140959650 frente al
margen de V03, y el leave-one-out -0.0004650203, que es casi nula.
Tampoco gana en la region larga, que es donde P63 midio el margen, y el margen tampoco mejora
ahi, luego el problema de MUY_LARGA no es de representacion ni de confianza.

**El perfil segmental es longitud disfrazada.** El mejor rasgo da
0.589847 de AUC frente a
0.600506 de contar tokens, luego queda
cerrado como rasgo.

**La razon de fondo.** El margen de V03 solo ya explica sus propios errores con AUC 0,770875
contra el acierto, y el mejor rasgo de evidencia interna llega a 0,605302. La estructura no
falta: esta dentro de la confianza de V03.

```text
the internal evidence distribution of long tweets is real, measured and reproducible,
and V03 already carries it inside its own confidence

V03 remains the best verified leaderboard result.
```
<!-- /P64 2026-09-29 -->

<!-- P65 2026-09-29 -->
## P65 (2026-09-29) - Representacion de largo alcance, alcance de operadores y jerarquia

Cuatro experimentos y una auditoria, trece figuras. Ninguno supera a V03, y la respuesta tiene
una forma que no estaba prevista.

**LA RESPUESTA: la senal existe y la adicion la rechaza.** Cinco candidatos de este proyecto
baten a los dos controles, el de capacidad y el de la confianza de V03, y ninguno mejora:

| candidato | contra el control de anchura | contra el margen de V03 | delta vs V03 |
|---|---:|---:|---:|
| P64-07 alcance de negacion | +0,0201566146 | +0,0088975231 | -0,0021826774 |
| P65-01 word (1,3) | +0.0596671770 | +0.0071281317 | -0,0039520688 |
| P65-03 cohesion documental | +0.0090835614 | +0.0083711943 | -0,0027090062 |

Lo que falta no es informacion: es sitio. Unas cientos de columnas compitiendo por el
presupuesto L2 de un espacio de veinte mil, y lo que la columna nueva necesita decir es mas
pequeno que lo que cuesta decirlo. Eso es una afirmacion sobre la ARQUITECTURA de la adicion,
y es falsable: predice que la misma evidencia fundida en el espacio de probabilidad, y no
concatenada en el de caracteristicas, sobrevive.

**EL GAP ANALYSIS CERRO MAS DE LO QUE ABRIO.** 6 abiertas y
10 cerradas, leidas en el codigo y no en el registro. Los skip-grams ya los
habia medido V6-05, con un vocabulario un 83 por ciento mayor y un resultado neutro. La
negacion y la cita las habian medido E2-07 y E2-08, y despues P64-06 y P64-07.

**EL CONTROL DE IMPLEMENTACION REPRODUCE A V6-05.** El rango de hueco 1 a 2, que es
exactamente el suyo, da
-0.0010515908 frente al -0,0001 que el
registro. Luego la implementacion de los pares con hueco es la misma, y los rangos que V6-05 no
alcanzo, de tres a cinco y de seis a diez tokens, tampoco aportan.

**Y LA LONGITUD DEL TWEET NO ES EL MECANISMO.** Un especialista ajustado en LARGA y MUY_LARGA
es BIT A BIT el mismo numero que un reentrenado sobre las mismas
714 filas sin ningun criterio de longitud:
+0.0000000000. Un negativo asi no es *faltan datos*,
que seria una excusa valida: dice que la especializacion no es el mecanismo, punto. Y el
especialista gana
+0.0212 en MUY_LARGA y
pierde -0.0257 en LARGA, que tiene
el triple de filas, luego hay una mejora local real que la metrica primaria rechaza.

```text
the information about long-range relations is present, measurable and not what V03 already
knows, and none of it fits through concatenation

the length of a tweet is not the mechanism either: a model retrained on the long rows with no
length criterion is bit for bit the specialist

V03 remains the best verified leaderboard result.
```
<!-- /P65 2026-09-29 -->

<!-- P66 2026-09-29 -->
## P66 (2026-09-29) - Late fusion: la evidencia de largo alcance en el espacio de scores

Dos experimentos y dos auditorias, trece figuras. Y es la primera vez que la senal de largo
alcance mueve el numero.

**LA RESPUESTA A LA PREGUNTA DE LA FASE: SI, Y ES ATRIBUIBLE AL CANDIDATO.**

| fusion | delta vs V03 | pliegues | brecha de atribucion |
|---|---:|---:|---:|
| MI_arbol + residual anidada | +0.0026743167 | 4/5 | +0.0041133588 |
| MI_arbol + residual con score barajado | -0.0014390421 | | |

La brecha de atribucion es la cifra que manda: el mismo mecanismo con el score del candidato
**barajado** pierde casi todo.

**Y MI_arbol es el unico que funciona porque es NO LINEAL.** Las correlaciones entre z_V03 y
z_C de los siete candidatos van de +0,9294 a +0,9988: son casi la misma puntuacion. El AUC del
residuo deja a cinco de siete por debajo de 0,5, luego su parte unica destruye informacion en
vez de anadirla. El de MI_arbol es el unico por encima del azar, y su modelo es el unico no
lineal de la lista. Si el error de V03 fuera una funcion lineal del error de MI_arbol, la
regresion residual no dejaria nada.

**EL MATIZ, Y VA PEGADO AL RESULTADO.** De las
53 celdas que la fusion rescata, solo
6 son errores de
alta confianza y las demas estan pegadas al umbral, luego es sobre todo **recalibracion**. Y el
bootstrap por 1415 grupos da IC95
[-0.0044311634, +0.0102108745], que
**incluye el cero**. Veredicto PROMISING, no FINALIST.

**EL INVENTARIO FUE EL PROBLEMA REAL.** Ningun script de P64 y P65 escribio sus
probabilidades a disco, y `features/` esta en `.gitignore`. Los scores se regeneraron y se
verificaron contra el numero de cada fase: 5 de 7 reproducen con diferencia de 1e-11, uno es
no verificable y uno tiene una diferencia declarada. El regenerador **se paro** cuando dos no
cuadraron.

**Y UNA CORRECCION QUE CAMBIO EL VEREDICTO DEL MEJOR RESULTADO.** `blend_prob` usaba siempre
la probabilidad REAL del candidato y nunca la barajada, luego su control de barajado salia
identico al real POR CONSTRUCCION, la brecha daba cero falso, y el mejor resultado de la fase
queda INVALIDATED por un motivo que no era el suyo. Corregido, ese mismo resultado pasa a
+0,0033 con brecha +0,0093.

```text
adding columns to a twenty thousand dimension space does not fit, and correcting a three
column score does, for the SAME evidence and with the SAME base model

the only measured exception is MI_arbol, and its distinguishing feature is that the model
producing it is NON LINEAR

and the caveat: it is mostly recalibration, because 6 of 53 rescues are high confidence and
the IC95 includes zero

V03 remains the best verified leaderboard result.
```
<!-- /P66 2026-09-29 -->

<!-- P67 2026-09-29 -->
## P67 (2026-09-29): el residuo de MI_arbol es MIXED, no una senal transferible

- Categoria del mecanismo: **MIXED_CON_ATRIBUCION_PARCIAL**.
- Reentreno anidado de un escalar por etiqueta: `-0.0029731513`, es decir la calibracion **no** explica el efecto.
- AUC del residuo de MI_arbol contra la etiqueta **fuera** de la banda: `0.579597`, sobre 0,5. Los otros seis candidatos caen por debajo.
- Sustitucion cruzada: `4 de 7` candidatos dan positivo con la misma operacion.
- Enriquecimiento de los rescates: `1.338302`.
- Efecto en celdas: `4.01` de 1483 filas.
- 1 de 12 tracks principales ejecutados, 0 de 4 reservas.
- 0 nuevas representaciones, 0 submissions, 0 Kaggle, 0 holdout, 0 test.
- Figuras: 6 en `figuras/p67/`.

> V03 remains the best verified leaderboard result.
<!-- /P67 2026-09-29 -->
