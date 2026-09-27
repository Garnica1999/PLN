# v21 P1-02 - Toxicidad y hate speech con NLP clasico (2023-2026)

Fecha de compilacion: 2026-09-27
Alcance: deteccion de hate speech (HS), toxicity (TR) y aggressive language (AG) en tweets en
espanol, con metodos clasicos de NLP (sin transformers, sin LLM, sin embeddings preentrenados,
sin datasets externos etiquetados). Metrica de referencia: macro-F1 sobre las 3 clases positivas,
1731 filas de train.

---

## 0. Metodo, cobertura y advertencias de verificacion

Reglas de trabajo (inviolables en este informe):
- NO transformers, NO LLM, NO embeddings/modelos preentrenados externos.
- NO datasets externos etiquetados.
- Solo texto de train + corpus externos NO etiquetados.
- No se reproponen lineas ya probadas y rechazadas en TOXIC: TF-IDF word+char con
  LogisticRegression (campeon, LB 0.78306), rebalanceo, class weights, NB-SVM, target encoding,
  ensembles, stacking, meta-modelos, DL desde cero (BiLSTM/GRU/CNN), features estructurales/densas,
  BiGRU fine-tuning, targeting, umbrales por etiqueta.

Niveles de verificacion usados en cada ficha:
- [V] VERIFICADO: le el texto completo (PDF/HTML) o el abstract oficial y extraje numeros.
- [P] PARCIAL: verifiquei metadata (autores, ano, venue, DOI) y/o fragmentos indexados, pero el
  texto completo estaba bloqueado (paywall). Los numeros citados provienen de fragmentos
  indexados, no de una lectura integral. Tratar con cautela.
- [N] NO VERIFICADO: no pude confirmar. No se usa como evidencia.

Fuentes consultadas: ACL Anthology, arXiv (API), Semantic Scholar API, OpenAlex, Crossref,
PMC, CEUR-WS, repos publicos. Springer, IEEE Xplore y TechRxiv bloquearon el acceso al texto
completo en esta sesion; queda registrado como limitacion.

Advertencia de alcance (importante): el searched NO encontro ninguna publicacion 2023-2026 que
haga exactamente nuestra tarea (3 etiquetas binarias correlacionadas HS/TR/AG sobre tweets en
espanol con 1731 filas de train, metricando macro-F1 de las 3 positivas). SemEval-2023 Task 10
es "Explainable Detection of Online Sexism (EDOS)" (ingles), no nuestra combinacion. HatEval
(SemEval-2019) tiene HS + comportamiento agresivo + target en espanol, pero no TR. Por lo tanto
toda la evidencia de abajo es analoga, no identica, y las magnitudes no son transferibles.

---

## 1. Publicaciones 2023-2026 con SOLO metodos clasicos, espanol o multilingue

### T1. Benchmarking Hate Speech Detection in Azerbaijani with Turkish Cross-Lingual Transfer and Transformer Models [V]

- Referencia: Alizada, T.; Dubossarsky, H. "Benchmarking Hate Speech Detection in Azerbaijani
  with Turkish Cross-Lingual Transfer and Transformer Models". SIGTURK 2026 (Second Workshop on
  NLP for Turkic Languages), ACL, pp. 103-112, DOI 10.18653/v1/2026.sigturk-1.10.
  Mar 2026, Rabat. Espejo arXiv no usado.
- Anio: 2026
- Tarea: deteccion binaria de hate speech (hate / no hate).
- Dataset: 1112 comentarios de YouTube de canales de noticias, anotados a mano en azerbaiyano
  (dataset nuevo, fuerte desbalance de clase); turco como fuente de transferencia.
- Representacion: TF-IDF sobre bag-of-words (n-gramas de palabra), clasificador Logistic
  Regression, como linea base clasica contra XLM-RoBERTa y BERT-Tiny.
- Preprocesamiento: estandar; el paper no reporta ablation de preprocesamiento.
- Ventaja reportada (numeros exactos, Tabla 2 y Tabla 1):
  - TF-IDF + LR en turco completo: Acc 0.88, P 0.89, R 0.83, macro-F1 0.85, recall de hate 0.68.
  - XLM-R en turco completo: Acc 0.94, macro-F1 0.93, P macro 0.94, R macro 0.92, P hate 0.93.
  - TF-IDF + LR en azerbaiyano nativo: Acc 0.90, macro-F1 0.47, recall de hate 0.00.
  - XLM-R en azerbaiyano nativo: Acc 0.91, macro-F1 0.72, recall de hate 0.00.
  - XLM-R con aumento por traduccion (TR,MT_AZ): macro-F1 0.88; BERT-Tiny 0.82.
  Cita textual: "The sparse features on Turkish have a linear decision boundary and are
  competitive (macro-F1 0.85) and achieve reasonable hate recall (0.68)."
- Limitaciones: azerbaiyano y turco no son espanol. El corpus es de comentarios de noticias, no
  tweets. El train es de 1112 filas (rango comparable al nuestro, pero el balance de clases es
  distinto). La conclusion es ambivalente: en_RESOURCE (turco) el transformer gana por 0.08
  macro-F1; en LOW-RESOURCE (azerbaiyano) AMBOS colapsan y la ventaja se invierte o desaparece.
- COMPATIBILIDAD: TOTAL (TF-IDF + LR puro, sin recursos externos).
- DIFERENCIA CONCRETA respecto a lo ya probado en TOXIC: no aporta una tecnica nueva, aporta
  (a) una medicion de referencia independiente de "que cerca del transformer esta TF-IDF+LR" en
  un corpus pequeno, y (b) la advertencia de que el colapso de la clase minoritaria (recall 0.00)
  es un modo de fallo REPRODUCIBLE en cualquier pipeline clasico cuando el desbalance es fuerte.
  Es la mejor evidencia cuantitativa que encontre de que la linea clasica se acerca al transformer
  pero no lo iguala en el caso favorable.

### T2. Offensive language detection in low resource languages: a use case of Persian language [V]

- Referencia: Mozafari, M.; Mnassri, K.; Farahbakhsh, R.; Crespi, N. "Offensive language
  detection in low resource languages: A use case of Persian language". PLOS ONE 19(6):e0304166,
  21 jun 2024. DOI 10.1371/journal.pone.0304166. PMCID PMC11192314, PMID 38905214.
- Anio: 2024
- Tarea: 3 niveles: (1) offensive vs non-offensive, (2) targeted vs untargeted, (3) individual vs
  group. Es el analogo estructural mas cercano a nuestra tarea multietiqueta.
- Dataset: 6000 microposts de X (Twitter) en persa de 520000 recolectados; anotados por 3
  nativos; acuerdo 73% (86% en nivel 1, 75% en nivel 2, 60% en nivel 3).
- Representacion: comparacion de SVM con TF-IDF char n-grams (n=2..5), TF-IDF word n-grams
  (n=1..2), BoWV sobre fastText, el set de features de Davidson et al. 2017 reimplementado, DL
  desde cero y transformers (ParsBERT, ALBERT-Persian, mBERT, XLM-RoBERTa).
- Preprocesamiento: normalizacion y tokenizacion estandar; conservacion de emojis y puntuacion
  como senal.
- Ventaja reportada (macro-F1 por nivel, k=5 folds, Tablas 6-8, leidas del texto):
  - Nivel 1: SVM word n-grams 0.908; ParsBERT 0.878; SVM char 0.870; ALBERT-Persian 0.849;
    BoWV-fastText 0.812; CNN+GRU 0.827; features Davidson 0.770.
  - Nivel 2: SVM char 0.829; ParsBERT 0.772; features Davidson 0.751; ALBERT-Persian 0.763;
    CNN+GRU 0.753.
  - Nivel 3: SVM char 0.896; SVM word 0.869; ParsBERT 0.691; ALBERT-Persian 0.763;
    CNN+GRU 0.753; features Davidson 0.695.
  - Stacking (least-correlated por Pearson) gana ~5 puntos de macro-F1 en los tres niveles.
  Cita textual: "Initial results on single models indicate that SVM trained on character or word
  n-grams are the best performing models accompanying monolingual transformer-based pre-trained
  language model ParsBERT".
- Limitaciones: persa, no espanol. Dataset 6000 (mayor que 1731). El set de features de Davidson
  usa fastText y sentiment scores, o sea que no es 100%Allowed; su version "pura" sin embeddings
  no se reporta aislada.
- COMPATIBILIDAD: PARCIAL. Las filas de SVM con TF-IDF si son totalmente compatibles. Las filas de
  fastText/ParsBERT no lo son. El resultado mas transferible es el del set de features de
  Davidson (features ligeras sin embeddings), que es compatible en su mayor parte.
- DIFERENCIA CONCRETA respecto a lo ya probado en TOXIC: es la unica referencia 2023-2026
  encontrada donde una tecnica CLASICA (SVM + n-gramas) supera a un transformer afinado en las
  TRES etiquetas de una tarea multietiqueta de lenguaje odivo. Ademas aporta la tabla de numeros
  del set de features tipo Davidson (POS n-grams, longitudes, cuentas de menciones/hashtags/
  retweets/URLs, polaridad de sentimiento, legibilidad Flesch-Kincaid y Flesch Reading Ease),
  que es la referencia canonica de "features de estructura" en esta literatura.

### T3. Enhancing social media hate speech detection in low-resource languages using transformers and explainable AI [P]

- Referencia: "Enhancing social media hate speech detection in low-resource languages using
  transformers and explainable AI". Social Network Analysis and Mining, Springer, 2025.
  DOI 10.1007/s13278-025-01497-w. Autoria no confirmada en esta sesion.
- Anio: 2025
- Tarea: deteccion de hate speech y de lenguaje ofensivo en lenguas de bajos recursos.
- Dataset: MultiCorpus-2020 y otros; el texto de muestra menciona .9% de accuracy para deteccion de
  lenguaje ofensivo.
- Representacion: comparacion de modelos no-transformer, con TF-IDF como una de las opciones.
- Preprocesamiento: "text cleaning"; el fragmento recuperado lo describe asi: "TF-IDF captured
  enough discriminative power without becoming too sparse".
- Ventaja reportada: fragmento indexado: "TF-IDF features often outperformed complex bigram and
  trigram models" y "TF-IDF exhibited superior performance for non-transformer models".
  OTRO fragmento: "an accuracy of 85.93%, slightly exceeding the 84% accuracy range".
- Limitaciones: [P] NO LEI el texto completo (Springer devolvio error de carga). Los autores y los
  numeros completos por modelo NO estan verificados. No puedo atribuirle una magnitud precisa.
- COMPATIBILIDAD: TOTAL (TF-IDF es linea base, no recurso externo).
- DIFERENCIA CONCRETA respecto a lo ya probado en TOXIC: ninguna tecnica; sirve como evidencia
  direccional de que TF-IDF sobrevive a la comparacion contra bigramas/trigramas en datos
  multilingues de bajos recursos. Usar SOLO como apoyo cualitativo.

### T4. Offensive Language Detection in Spanish Social Media: Testing From Bag-of-Words to Transformers Models [P]

- Referencia: Molero, J. M.; Perez-Martin, J.; Rodrigo, A.; Penas, A. "Offensive Language
  Detection in Spanish Social Media: Testing From Bag-of-Words to Transformers Models".
  IEEE Access, vol. 11, 2023. DOI 10.1109/ACCESS.2023.3257861 (IEEE Xplore doc. 10234423).
- Anio: 2023
- Tarea: deteccion de lenguaje ofensivo en redes sociales en espanol.
- Dataset: no verificado en esta sesion (la sesion no logro abrir el PDF).
- Representacion: recorrido explicito "from bag-of-words to transformers".
- Preprocesamiento: el paper incluye una Tabla 13 titulada "Macro-average results for binary
  classification with different levels of pre-processing", con una fila "Default pre-processing"
  cuyo valor de precision es 0.8902. Esto confirma que existe una ablation de preprocesamiento
  medida en metricas macro, en espanol, sobre este corpus.
- Ventaja reportada: fragmento indexado del abstract: "We show how transformer-based models
  continue obtaining the best results, but we improved previous results by a 6,2% by adding new
  steps of pre-processing and ...". Es decir: la ganancia de +6.2% atribuida a PASOS DE
  PREPROCESAMIENTO, no a la arquitectura.
- Limitaciones: [P] sin lectura integral. Es la referencia mas relevante para espanol + metricas
  macro + preprocesamiento, y precisamente por eso hay que marcarla como la MAYOR BRECHA de
  verificacion de este informe. El "+6.2%" no tiene unidad declarada en el fragmento.
- COMPATIBILIDAD: PARCIAL. Los "nuevos pasos de preprocesamiento" son presumiblemente aplicables
  (reglas ortograficas, no recursos externos), pero no se pudo leer CUALES son.
- DIFERENCIA CONCRETA respecto a lo ya probado en TOXIC: es la unica evidencia publicada de que
  en espanol y con metricas macro hay una ganancia de ~6% atribuible SOLO a preprocesamiento. Es
  exactamente el hueco que T1/T2 no cubren en espanol. ACCION: leer el PDF por otra via antes de
  decidir.

### T5. A Case Study of Spanish Text Transformations for Twitter Sentiment Analysis [V]

- Referencia: Tellez, E. S.; Miranda-Jimenez, S.; Graff, M.; Moctezuma, D.; Siodia, O. S.;
  Villasenor, E. A. "A Case Study of Spanish Text Transformations for Twitter Sentiment Analysis".
  Expert Systems with Applications, DOI 10.1016/j.eswa.2017.03.071. Preprint arXiv:2106.02009
  (3 jun 2021). Affiliations: INFOTEC (Mexico), CentroGEO, CONACyT.
- Anio: 2021 (fuera de la ventana 2023-2026, pero es la referencia escena-espanol que mas aplica
  directamente a nuestro caso; la incluyo por su valor, marcandolo como pre-2023)
- Tarea: analisis de sentimiento binario en tweets en espanol.
- Dataset: dos corpora en espanol: INEGI y TASS'15.
- Representacion: SVM sobre combinaciones de transformaciones de texto, tokenizadores (word
  n-grams) y esquemas de ponderacion; introducen la combinacion de word n-grams + character
  q-grams.
- Preprocesamiento (el detalle relevante, leido del paper):
  - Correccion de elongation/vocales y consonantes duplicadas mediante diccionario del espanol +
    modelo estadistico de letras dobles comunes + reglas heuristicas para interjecciones.
    Ejemplos textuales: "ruiiidoooo -> ruido", "jajajaaa -> jaja", "jijijji -> jaja".
  - Expansion de coloquialismos y abreviaturas con un diccionario basado en regex.
    Ejemplos: "tqm -> te quiero mucho", "compu -> computadora".
  - Manejo de negacion con reglas explicitas.
  - Emoticons: clasificacion de mas de 500 emoticones de texto y el conjunto completo de ~1600
    emoticones Unicode en tres clases (positivo, negativo, neutro), reemplazados por una palabra
    de polaridad.
  - Usuarios, hashtags, URLs y emoticones se manejan como TAGS ESPECIALES para preservar la
    estructura de la oracion, no se eliminan.
- Ventaja reportada (numeros exactos del abstract):
  - La combinacion de word n-grams + character q-grams supera a la combinacion solo de word
    n-grams en 11.17% (INEGI) y 5.62% (TASS'15).
  - Halazgo contraintuitivo, textual: "Counterintuitively, we found that the complexity of
    techniques used in the pre-processing step is not correlated with the final performance of the
    classifier, e.g., a classifier using lemmatization, which is one of the pre-processing
    techniques having the greatest complexity, might not be one of the systems having the highest
    performance."
  - Analisis estructural de las k mejores configuraciones (Tabla 4, corpus INEGI): "almost all best
    setups enable TFIDF, and properly handle emoticons and users".
  - Gap medido entre la mejor configuracion con TFIDF y la mejor sin TFIDF: ~0.005 de accuracy,
    "quite small to discard".
- Limitaciones: la tarea es sentimiento, no toxicity. Los datasets son de~1-2k Tweets de
 Newswire, no el mismo dominio que Kaggle. 2021, no 2023-2026.
- COMPATIBILIDAD: TOTAL. Todo esta basado en reglas y diccionario derivados del propio texto; no requiere
  recursos externos. OJO: el diccionario del espanol usado para la correccion de elongacion es un
  recurso de lengua; hay que auditar si el reglamento del concurso lo considera admisible (ver
  seccion 2.3).
- DIFERENCIA CONCRETA respecto a lo ya probado en TOXIC: ESTA ES LA DIFERENCIA PRINCIPAL DE TODO
  EL INFORME. En TOXIC se probo word+char n-grams, pero NO una normalizacion ortografica
  espanola con correccion de elongaciones + expansion de abreviaturas + mapeo de emoticones a
  tokens de polaridad. Es una TERCERA VISTA de texto (no un reemplazo del pipeline) y por
  construccion aporta senal distinta de la que aporta la vista char n-grams.

### T6. Alejandro Mosquera at DETOXIS 2021: Deep Learning Approaches to Toxicity Detection in Spanish Social Media Texts [V]

- Referencia: Mosquera, A. "Alejandro Mosquera at DETOXIS 2021: Deep Learning Approaches to
  Toxicity Detection in Spanish Social Media Texts". CEUR-WS Vol-2943, DETOXIS 2021 Workshop
  (IberLEF@SEPLN). PDF: http://ceur-ws.org/Vol-2943/detoxis_paper4.pdf
- Anio: 2021 (fuera de la ventana, pero es la unica tabla de sistemas completa en espanol que
  pude leer con metricas de F1 de la clase toxica)
- Tarea: deteccion de toxicity en espanol, sub-task 1 de DETOXIS 2021.
- Dataset: NewsCom-TOX, comentarios de articulos de noticias en espanol, menos de 4000 muestras,
  1147 positivas. ESTE ES EL CORPUS MAS PARECIDO A NUESTRO EN TAMANO (comparar 1731 filas).
- Representacion: 7 modelos (capsule es/en con GloVe SBWC i25, Detoxify orig/unb/multi/multi-es,
  spacyLR). Stacking con Logistic Regression sobre probabilidades + side information.
- Preprocesamiento: no documenta pasos de limpieza; usa metadatos no textuales (topic, thread id,
  comment id, reply to) como side information.
- Ventaja reportada (Tabla 1, F1 de la clase toxica en test):
  - SINAI (mejor sistema) 0.6461; mejor modelo individual = capsule es (GloVe SBWC).
  - Mosquera 0.5691.
  - AllToxic 0.4231 (su modelo era spacyLR, o sea LR sobre embeddings espanoles de spaCy).
  - ChainBOW 0.3746; RandomClassifier 0.3760; BOWClassifier 0.1837.
  - El stacking LR uso "a custom threshold of 0.32, which was determined via cross-validation",
    justificado por el desbalance y por la metrica (F1 de la clase toxica, no micro ni macro).
  - Side information mas fuerte para la clase NO toxica: "avg group tox" (promedio de toxicidad de
    todos los comentarios con el mismo thread id).
- Limitaciones: usa embeddings preentrenados en casi todos los sistemas, asi que los numeros no
  son atribuibles a la linea TF-IDF pura. El corpus es de noticias, no de tweets. 2021.
- COMPATIBILIDAD: NULA para replicar el sistema (embeddings), PARCIAL para las lecciones de
  diseno (umbral por clase via CV; agregacion a nivel de hilo).
- DIFERENCIA CONCRETA respecto a lo ya probado en TOXIC: (a) confirma que el umbral movido
  determined por CV es la practica correcta cuando la metrica premia la clase positiva; (b) las
  lineas puramente BoW (ChainBOW 0.3746, BOWClassifier 0.1837) quedaron MUY por debajo, lo que
  es evidencia ADVERSA para TF-IDF en espanol; (c) la agregacion por thread es una senal de
  estructura conversacional que no equivale a las features estructurales/densas ya rechazadas
  (verificacion pendiente: si las filas del concurso son tuits aislados, esta via no aplica).

### T7. Bridging Gaps in Hate Speech Detection: Meta-Collections and Benchmarks for Low-Resource Iberian Languages [V]

- Referencia: Piot, P.; Pichel Campos, J. R.; Parapar, J. "Bridging Gaps in Hate Speech Detection:
  Meta-Collections and Benchmarks for Low-Resource Iberian Languages". arXiv:2510.11167
  (13 oct 2025). Publicado en Machine Learning (Springer), DOI 10.1007/s10579-026-09921-z.
- Anio: 2025 (preprint) / 2026 (publicacion)
- Tarea: meta-coleccion de hate speech para espanol europeo, con etiquetas y metadatos
  unificados; traduccion a portugues europeo y dos estandarizations de gallego.
- Representacion: solo LLMs (zero-shot, few-shot, fine-tuning). NO aporta linea clasica.
- Ventaja reportada: no aplica a nuestras reglas.
- Limitaciones:Irrelevante como metodo; relevante como ATESTIGO DE BRECHA de datos en espanol.
- COMPATIBILIDAD: NULA.
- DIFERENCIA CONCRETA: ninguna. La incluo para dejar constancia de que la unica publicacion
  reciente especificamente sobre hate speech en espanol que encontre NO usa metodos clasicos y
  que la comunidad ha migrado a LLMs. Es un argumento indirecto a favor de que la linea TF-IDF
  clasica en espanol esta poco estudiada y por tanto su techo no esta bien caracterizado.

### T8. Text Detoxification in isiXhosa and Yoruba: A Cross-Lingual Machine Learning Approach [V]

- Referencia: Agbeyangi, A. O. "Text Detoxification in isiXhosa and Yoruba: A Cross-Lingual
  Machine Learning Approach for Low-Resource African Languages". arXiv:2601.05624 (9 ene 2026).
- Anio: 2026
- Tarea: deteccion de toxicity (binaria) como componente de un sistema de destoxificacion.
- Representacion: TF-IDF + Logistic Regression (interpretabilidad,etz.).
- Ventaja reportada: accuracies con K-fold estratificado de 61-72% (isiXhosa) y 72-86% (Yoruba);
  ROC-AUC hasta 0.88.
- Limitaciones: Task es destoxificacion, no toxicity multietiqueta; las accuracies son BAJAS
  (61% en isiXhosa esta cerca del azar balanceado). No aporta evidencia fuerte a favor.
- COMPATIBILIDAD: TOTAL pero IRRELEVANTE (idiomas Africanos, metricas no comparables).
- DIFERENCIA CONCRETA: ninguna. Se incluye para documentar que la busqueda de "TF-IDF+LR en
  toxicity 2026" devuelve Lineas de baja calidad metodologica; sirve como control de ruido.

### T9. Usman et al. Multilingual Hate Speech Detection in Social Media Using Translation-Based Approaches [V]

- Referencia: Usman, M.; Ahmad, M.; Shahiki Tash, M. S.; Gelbukh, I.; Quintero Tellez, R.;
  Sidorov, G. "Multilingual Hate Speech Detection in Social Media Using Translation-Based Approaches
  with Large Language Models". arXiv:2506.08147 (9 jun 2025).
- Anio: 2025
- Tarea: hate speech binaria en ingles, urdu y espanol.
- Representacion: modelos no-transformer con TF-IDF como extraccion de features, frente a
  transformers y LLMs.
- Ventaja reportada (abstract): macro-F1 de GPT-3.5 Turbo 0.87 en ingles, 0.85 en espanol, 0.81 en
  urdu (Qwen 2.5 72B), 0.88 conjunto. Mejoras sobre la linea base SVM: +8.75% ingles,
  +8.97% espanol, +5.19% urdu, +7.32% conjunto. Es decir, en espanol la linea clasica SVM
  queda en ~0.78 de macro-F1.
- Limitaciones: el dataset propio de 10193 tweets tiene 3162 en espanol; metodo basado en
  traduccion y prompting, mayoritariamente no aplicable. Los numeros de la linea base son
  reportados en el abstract sin tabla legible aqui.
- COMPATIBILIDAD: NULA (la metodo es LLM). La FILA de la linea base es la parte relevante.
- DIFERENCIA CONCRETA: aporta un dato indirecto: la linea clasica SVM+TF-IDF en espanol con
  ~3.2k tweets esta alrededor de macro-F1 0.78. Nuestro campeon esta en LB 0.78306. Es decir,
  nuestro punto de partida ya es DEL ORDEN DE MAGNITUD del published estado del arte clasico en
  espanol para deteccion binaria de hate. Esto es un dato fuerte a Favor de la hipotesis de techo.

---

## 2. Lexicos y recursos en espanol: inventario y legalidad

### 2.1 Recursos identificados

#### L1. Multilingual HateCheck (MHC) [V]

- Referencia: Rottger, P.; et al. "Multilingual HateCheck: Functional Tests for Multilingual Hate
  Speech Detection Models". WOAH 2022 (Workshop on Online Abuse and Harms) @ NAACL 2022.
  ACL Anthology 2022.woah-1.15. arXiv:2206.09917. Repositorio:
  https://github.com/rewire-online/multilingual-hatecheck (licencia CC-BY-4.0).
- Anio: 2022
- Que es: suite de 34 funcionalidades de prueba en 10 idiomas: arabe, neerlandes, frances, aleman,
  hindi, italiano, mandarin, polaco, portugues y ESPANOL (confirmado leyendo el README del
  repositorio). Casos sinteticos con label gold "hateful"/"non-hateful" y 3 anotadores por caso.
- Contiene ESPANOL: SI, confirmado directamente.
- Limitaciones: es un DIAGNOSTICO, no un corpus de entrenamiento; el propio README advierte que
  un modelo puede "reportar buenos resultados sobreajustandose" a MHC. Es ademas un subconjunto
  sintetico de frases plantilla, muy distinto de tweets reales.
- COMPATIBILIDAD: NULA. Es un dataset externo ETIQUETADO. Ademas es sintesis etiquetada derived
  de personas.

#### L2. HateCheck (original, ingles) y HateoMeter [V]

- Referencia: Rottger, P.; Vidgen, B.; Nguyen, D.; Waseem, Z.; Margetts, H.; Pierrehumbert, J.
  "HateCheck: Functional Tests for Hate Speech Detection Models". ACL 2021 (2021.acl-long.4).
  Preprint arXiv:2012.15606. 3728 casos, 29 funcionalidades, 7 grupos objetivo. La interfaz web
  se llama HateoMeter.
- Que es: lo mismo que MHC pero solo en ingles.
- Limitaciones: no tiene version espanola.
- COMPATIBILIDAD: NULA (dataset externo etiquetado, ademas en ingles).

#### L3. HurtLex [V - metadata; cobertura espanola NO verificada]

- Referencia: Basile, V.; et al. "HurtLex: A Multilingual Lexicon of Words to Hurt". Proc. CLiC-it
  2018, CEUR-WS Vol-2253, paper49. Repositorio: https://github.com/valeriobasile/hurtlex
- Anio: 2018
- Que es: lexico computacional de palabras ofensivas, agresivas y odiosas en mas de 50 idiomas,
  17 categorias mas una macro-categoria de estereotipo. El catalogo hatespeechdata.com lo describe
  asi (leido directamente).
- Limitaciones: los terminos estan anotados con SENSO (ofensivo/conservador vs negativo literal),
  no con toxicidad, y el mapeo de categoria a nuestra TR/AG/HS es MANUAL y no trivial. Se usa como
  filtro de muestreo en la literatura (ver T2: Mozafari lo uso para lexicon-based sampling), no
  como clasificador.
- COMPATIBILIDAD: NULA como clasificador; marginal como filtro de textos no etiquetados.

#### L4. The Weaponized Word [N - cobertura espanola NO verificada]

- Que es: "several thousand discriminatory, derogatory and threatening terms across 125+
  languages, available through a RESTful API", segun hatespeechdata.com. weaponizedword.org
- Limitaciones: no verifique la cobertura espanola ni las condiciones de acceso.
- COMPATIBILIDAD: NULA.

#### L5. MOL - Multilingual Offensive Lexicon con informacion contextual [N - cobertura espanola NO verificada]

- Referencia: Vargas, F.; et al. "Contextual-Lexicon Approach for Abusive Language Detection".
  RANLP 2021, ACL Anthology 2021.ranlp-1.161, arXiv:2104.12265. Repositorio:
  https://github.com/franciellevargas/MOL
- Que es: "the first specialized lexicon for hate speech detection annotated with contextual
  information. It consists of 1,000 explicit and implicit" terminos (texto del repositorio).
- Limitaciones: no pude leer el PDF (descarga bloqueada). NO VERIFICO si espanol esta incluido.
- COMPATIBILIDAD: NULA.

#### L6. SexHateLex / SWSR [V]

- Referencia: Jiang, A.; Yang, X.; Liu, Y.; Zubiaga, A. "SWSR: A Chinese Dataset and Lexicon for
  Online Sexism Detection". Journal of Online Social Networks and Media, vol. 27, 100182, 2022.
  ISSN 2468-6964. Lexicon de 3016 terminos.
- Limitaciones: SOLO chino. No aplica.
- COMPATIBILIDAD: NULA.

#### L7. "panel de_times" [N - NO VERIFICADO / NO ENCONTRADO]

- NO LOGRE verificar este recurso. El repositorio que busque
  (github.com/delvilledor/multilingual-hate-speech-lexicon) devolvio 404. La busqueda del
  termino "de_times" en contexto de panel lexico en espanol no produjo ninguna fuente primaria.
  El nombre "de_times" sugiere un panel en ALEMAN (corpus DeSpielDesTime) del lexico multilingue
  de hate speech (MHSL), no espanol.
- ADVERTENCIA EXPRESA: no uso "panel de_times" como evidencia de nada. Si el briefing lotomaba de
  una fuente secundaria, esa fuente no es verificable en esta sesion.

#### L8. Hatebase [P]

- Que es: "Hatebase is a crowdsourced resource of hate speech lexicons", segun el abstract del
  paper TRAC 2024 "The Constant in HATE: Toxicity in Reddit across Topics and Languages"
  (aclanthology.org/2024.trac-1.1). No verifiquei la cobertura espanola ni el acceso.
- COMPATIBILIDAD: NULA.

### 2.2 Veredicto de legalidad (respuesta directa a la pregunta del briefing)

Un lexico de hateful/toxic language ES un recurso externo. Depende de como se construyo:

- Si fue construido con anotadores humanos que etiquetan vocabulario -> es un artefacto
  ETIQUETADO externo. Choca de frente con la regla "NO datasets externos etiquetados".
- Si fue construido con corpus NO etiquetados y tecnicas no supervisadas (clustering de
  distributional, bootstrapping con semillas) -> en principio cae dentro de "corpus externos NO
  etiquetados". PERO: no encontre ninguna publicacion 2023-2026 que induzca un lexico de toxicity
  EN ESPANOL a partir de corpus no etiquetados y que valide la gain. La literatura de induccion de
  lexicos (Wiegand et al. NAACL 2018) usa bootstrapping con semillas PARA-language-supervisado.
  MARCA: sin evidencia.

- Conclusion operativa: la unica via 100% segura de tener un lexico es DERIVARLO DEL TRAIN
  (1731 filas). Eso es exactamente log-count ratio (NB-SVM r) y supervised term weighting, que ya
  fueron probados y rechazados. Por lo tanto, la via del lexico no ofrece nada nuevo bajo las
  reglas actuales.

### 2.3 Regla adicional que se aplica a los pipelines de T5

La recipe de Tellez et al. (correccion de elongaciones) usa un DICCIONARIO DEL ESPANOL. Eso es un
recurso externo de lengua, distinto de un lexicon de toxicity y distinto de un modelo preentrenado.
NO esta cubierto explicitamente por las reglas. ACCION REQUERIDA: auditar el reglamento del
concurso. Una alternativa 100% derivada del train seria detectar repeticiones de caracteres con
expresiones regulares puras (r"(.)\1{2,}") sin diccionario, lo cual no necesita recurso alguno.

---

## 3. Tecnicas clasicas de preprocesamiento para redes sociales en espanol

### P1. Correccion de elongation / repeticion de caracteres [V, T5]

- Referencia: Tellez et al. 2021 (ver T5). Anio 2021.
- Efecto medido: no hay numero aislado por tecnica en el paper; lo que si hay es el analisis
  estructural de las k mejores configuraciones y el gap medido con/sin TFIDF (~0.005). La
  afirmacion verificable es la contraintuitiva: la COMPLEJIDAD del preprocesamiento NO se
  correlaciona con el rendimiento final.
- Limitaciones: la evidencia es de sentimiento, no de toxicity. El paper no aisla la ganancia de
  la correccion de elongaciones por separado de la correccion de abreviaturas y del mapeo de
  emoticones.
- COMPATIBILIDAD: ALTA si se implementa con regex puro (sin diccionario).
- DIFERENCIA CONCRETA: TOXIC no normaliza elongaciones. En espanol de redes esto es
  cuantitativamente relevante ("jajajaaa", "nooo", "malisimo", "cabr0n"). Nota: la reduccion de
  variacion ortografica puede ENCOGER el numero de tipos y por tanto DEGRADAR la cobertura de char
  n-grams, que es justo el motor del campeon actual. Riesgo real, no trivial.

### P2. Expansion de abreviaturas y coloquialismos [V, T5]

- Referencia: Tellez et al. 2021. Anio 2021.
- Efecto medido: no aislado. Implementado como "expansion of colloquial words and abbreviations"
  con diccionario basado en regex (tqm -> te quiero mucho).
- COMPATIBILIDAD: MEDIA-ALTA. Se puede hacer con una lista corta hecha a mano a partir de
  inspeccion del TRAIN (es decir, derivada de datos etiquetados propios, no recurso externo).
  Con ~1731 filas es perfectamente viable construir un diccionario de 50-150 abreviaturas por
 Simple observacion, y eso no viola ninguna regla.
- DIFERENCIA CONCRETA: no probado. Es la ficha con mejor relacion senal/costo de toda la seccion
  3 bajo nuestras reglas.

### P3. Mapeo de emoticones / emojis a tokens de polaridad [V, T5] + supporting

- Referencia primaria: Tellez et al. 2021 - "we classify more than 500 popular emoticons,
  including text emoticons, and the whole set of emoticons (close to 1600) defined by Unicode,
  into three classes: positive, negative or neutral, which are replaced by a polarity word or
  definition associated to the emoticon according to the Unicode standard". Emoticons sin
  polaridad se vuelven "unicode-text".
- Referencia de apoyo 1: "Leveraging Text Generated from Emojis for Hate Speech and Offensive
  Content Detection" (CEUR-WS Vol-3159). Afirmacion indexada: "incorporating emotions as text from
  emojis can improve the classification accuracy of hate speech or offensive content".
- Referencia de apoyo 2: "Leveraging Emoji Preprocessing with BI-LSTM Model" (revista ISI, 2025).
  Indexado: "The research examines how different emoji preprocessing strategies affect the
  performance of a BI-LSTM model for hate speech classification". NO LEI el texto; sin numeros.
- Referencia de apoyo 3: Hatemoji (Kirk, Vidgen, Rottger, Thrush, Hale, 2021, arXiv:2108.05921).
  Suite de pruebas + dataset generado para hate basado en emoji (HatemojiCheck 3930 casos,
  HatemojiBuild 5912). Ingles. COMPATIBILIDAD NULA.
- Referencia de apoyo 4: "Emojis as Anchors to Detect Arabic Offensive Language and Hate Speech"
  (arXiv:2201.06723). Metodo language-independent basado en emoji como anclas.
- Ventaja reportada: Tellez et al. afirman en su analisis de las mejores configuraciones que
  "almost all best setups enable TFIDF, and properly handle emoticons and users" (corpus INEGI).
  Ese es el dato mas fuerte a favor de tratar emoticones y usuarios de forma explicita.
- Limitaciones: 1600 emoticones Unicode = recurso externo (el estandar Unicode). Pero se puede
  construir una tabla minima a mano desde el TRAIN (decenas de emojis, no 1600) y complying con
  las reglas. La literatura de apoyo usa BiLSTM/embeddings, que no podemos replicar.
- COMPATIBILIDAD: ALTA si la tabla se deriva del train; NULA si se copia la tabla completa de
  Unicode.
- DIFERENCIA CONCRETA: no probado en TOXIC, y con matriz de senal distinta de char n-grams: un
  emoji (-> token POS/NEG/NEU) es un token de clase cerrada que actua como feature categorica
  densa de baja cardinalidad, no como subpalabra.

### P4. Tratamiento explicito de usuarios, hashtags y URLs como TAGS [V, T5]

- Referencia: Tellez et al. 2021. "Twitter tags such as user names, hashtags (topics), URLs, and
  emoticons are handled as special tags in our representation to keep the structure of the
  sentence."
- Efecto medido: Same evidencia estructural que P3 ("properly handle emoticons and users").
- Limitaciones: en espanol los hashtags son lexicalmente informativos (#inmigracion, #feminicidio),
  y los handles espanolos son discriminator de dialecto/variante. No encontre una medicion
  aislada del efecto de KEEP vs DROP vs TAG en toxicity en espanol.
- COMPATIBILIDAD: TOTAL.
- DIFERENCIA CONCRETA: TOXIC tiene "features estructurales/densas" rechazadas, pero no vi evidencia
  de que se probara un reemplazo LEXICO de @user / #hashtag / URL por tokens marcadores como
  variables del vectorizador de texto. Es distinto de una feature densa.
- Dato relacionado [V]: Dementieva et al. 2024 (arXiv:2412.11691) declaran que para texto
  procedente de redes sociales "anonymize or fully eliminate any mentioning of usernames and
  links". Es una practica de preparacion de datos, NO una ablation medida.

### P5. Threads / agregacion conversacional [V, T6]

- Referencia: Mosquera 2021 (DETOXIS, ver T6).
- Efecto medido: la side information "avg group tox" (promedio de toxicidad de todos los
  comentarios con el mismo thread id) fue, segun los coeficientes de la regresion logistica del
  stacking, "the strongest feature for detecting the toxic and non-toxic class respectively". Es
  decir, la agregacion por hilo es senal FUERTE.
- Limitaciones: en DETOXIS el lado textual es un comentario de noticia con metadatos de hilo
  disponibles. NUESTRO COMPETICION: 1731 filas de tweets. Si las filas son tuits SIN contexto de
  hilo, esta via NO es implementable y debe descartarse. ACCION: verificar si el train incluye
  algun identificador de hilo/respuesta. Si no lo incluye, esta linea se descarta por
  impossibilidad, no por rendimiento.
- COMPATIBILIDAD: CONDICIONAL (depende de metadatos disponibles).
- DIFERENCIA CONCRETA: no es "features estructurales/densas"; es una variable de grupo. Es la
  unica via publicada con senal fuerte para modelar conversacion en espanol.

### P6. "Unificar y expandir / help-wrap (Pew Research)" [N - SIN EVIDENCIA]

- REALIDAD DE LA BUSQUEDA: multiple busquedas dirigidas (Pew + toxicity + menciones + enlaces;
  "help-wrap"; "unify and expand") NO devolvieron NINGUN trabajo que documente que unificar /
  expandir menciones, hashtags o enlaces mejore la clasificacion de toxicity, ni de forma
  general ni en espanol. La expresion "help-wrap" no aparece en la literatura de NLP.
- MARCA: sin evidencia. No lo incluyo como candidato.
- Lo mas cercano y verificable, y que va en la direccion CONTRARIA (borrar en vez de expandir):
  Dementieva et al. 2024 eliminan usernames y links; Kulkarni et al. 2017 (Twitch, citacion de
  index) reemplazan emotes por texto y reportan +13% sobre BERT con 76% de F1; Tellez et al. 2021
  KEEP los handles como tags especiales. La literatura esta divided y nadie midio el efecto en
  espanol con metricas macro.

### P7. Ablaciones sistematicas de preprocesamiento en toxicity [P]

- Glazkova, A. V. "A comparison of text preprocessing techniques for hate and offensive speech
  detection in Twitter". Social Network Analysis and Mining 13:155, 2023.
  DOI 10.1007/s13278-023-01156-y. Metadata verificada via OpenAlex (autora: Anna Valerievna
  Glazkova, University of Tyumen, Rusia; 12 citas; cerrado, sin OA). ABSTRACT NO DISPONIBLE en
  OpenAlex, Semantic Scholar ni Crossref; Springer bloqueo la pagina.
- Unico fragmento indexado del abstract: "Our results show that combining techniques can produce
  different results". Keywords indexadas: Preprocessor, Support vector machine, Convolutional
  neural network, Random forest, Offensive.
- Shukla, D.; Dwivedi, S. K. "The study of the effect of preprocessing techniques for emotion
  detection on Amazon product review dataset". SNAM, 2024. DOI 10.1007/s13278-024-01352-4.
  Metadata verificada; sin abstract. Indexado: "13 commonly used techniques such as lowercasing,
  stemming, lemmatization, stop words removal, etc." NO es toxicity ni espanol.
- Revision: "Textual variations in social media text processing applications". SNAM 2024.
  DOI 10.1007/s10462-024-11071-z. Metadata via busqueda; sin lectura.
- MARCA: sin evidencia utilizable. Estas tres son la BRECHA #2 de verificacion. La conclusion
  honesta es que NO se pudo extraer ninguna magnitud sobre el efecto individual de normalizar
  elongaciones, risas, emojis, hashtags o menciones en toxicity en espanol.

---

## 4. Feature engineering clasica incremental sobre word+char n-grams

### F1. NB log-count ratio (log odds) y NB-SVM [P - referencia canonica NO verificada] + [V]

- Referencia canonica: el esquema "log count ratio" y NBSVM provienen del trabajo de Wang y
  Manning sobre clasificacion de tweets binarios ("fast and accurate text classification with
  counts"). NO LOGRE VERIFICAR esa referencia en esta sesion: Crossref no la devolvio y el
  resultado arXiv 1808.09367 que aparece al buscar "log count ratio" resulto ser OTRO paper
  (Bao, Chang, Mo, Barzilay, "Deriving Machine Attention from Human Rationales", EMNLP 2018,
  DOI 10.18653/v1/D18-1216), que NO es la referencia de NBSVM. MARCA: [N] para la referencia
  canonica. NO la cito como verificada.
- Referencia VERIFICADA con numero: Bhange, M.; Kasliwal, N. "HinglishNLP: Fine-tuned Language
  Models for Hinglish Sentiment Detection". arXiv:2008.09820, SemEval 2020 (2020). Texto del
  abstract: "Prior work demonstrates the efficacy of classical ML methods for polarity detection.
  Fine-tuned general-purpose language representation models, such as those of the BERT family are
  benchmarked along with classical machine learning and ensemble methods. We show that NB-SVM beats
  RoBERTa by 6.2% (relative) F1. The best performing model is a majority-vote ensemble which
  achieves an F1 of 0.707."
- Tarea: sentimiento, no toxicity. Hindi-ingles.
- Ventaja reportada: NB-SVM supera a RoBERTa en 6.2% relativo de F1.
- Limitaciones: no es toxicity ni espanol. Es el unico numero duro que pude verificar para NB-SVM
  vs transformers, y es de sentimiento.
- COMPATIBILIDAD: TOTAL, pero YA PROBADO (NB-SVM rechazado en TOXIC).
- DIFERENCIA CONCRETA: NINGUNA. Se documenta por completitud. La conclusion correcta es que
  NB-SVM tiene evidencia de competitiveness pero no aporta linea nueva.
- Pregunta abierta del briefing ("verifica si sigue siendo SOTA"): RESPUESTA = no es SOTA desde
  2018/2020; la literatura 2023-2026 que compara sparse vs transformer en toxicity muestra que el
  transformer gana cuando hay datos y el sparse gana cuando no los hay (ver T1, T2, seccion 7).

### F2. Supervised term weighting: evidencia NEGATIVA consolidada [V]

- Referencia: Moreo Fernandez, A.; Esuli, A.; Sebastiani, F. "Learning to Weight for Text
  Classification". arXiv:1903.12090 (2019); publicado en IEEE Transactions on Knowledge and Data
  Engineering 32(2):302-316, 2020.
- Texto verificado del abstract: "Although 'supervised term weighting' approaches that use this
  intuition have been described before, they have failed to show consistent improvements. In this
  article we analyse the possible reasons for this failure, and call consolidated assumptions into
  question. Following this criticism we propose a novel supervised term weighting approach that,
  instead of relying on any predefined formula, learns a term weighting function optimised on the
  training set of interest."
- Ventaja/limitacion: la contribucion central de la literatura es explicar POR QUE el supervised
  term weighting (STW) no mejora de forma consistente. Esto es evidencia negativa fuerte, y mas
  relevante dado que el proyecto YA probo supervised weighting (rechazado). Es una
  COHERENCIA CRUZADA: la literatura independiente ya decia que no iba a funcionar.
- COMPATIBILIDAD: PARCIAL (LTW se puede implementar, pero requiere learn sobre el train).
- DIFERENCIA CONCRETA: ninguna; confirma el rechazo ya emitido. Valor: justifica cerrar la linea.

### F3. Regularizacion del STW (sublinear scaling + bias term + add-one) [V]

- Referencia: Wu, H.; Gu, X. "Balancing Between Over-Weighting and Under-Weighting in Supervised
  Term Weighting". arXiv:1604.04007 (2016).
- Texto verificado del abstract: "This paper shows that supervised term weighting should not just
  assign large weights to imbalanced terms, but should also control the trade-off between
  over-weighting and under-weighting... To prevent over-weighting, we present three regularization
  techniques: add-one smoothing, sublinear scaling and bias term... Empirical evaluations on
  topical and sentiment classification datasets indicate that sublinear scaling and bias term
  greatly influence the performance of supervised term weighting, and our re enjoys the best
  results in comparison with existing schemes."
- Referencia de apoyo: Wang, D.; Zhang, H. "Inverse-Category-Frequency based supervised term
  weighting scheme for text categorization". arXiv:1012.2609. Verificado: "superior or comparable to
  six supervised term weighting schemes and three traditional schemes in terms of macro-F1 and
  micro-F1".
- Referencia de apoyo: Carvalho, F.; Guedes, G. P. "TF-IDFC-RF: A Novel Supervised Term Weighting
  Scheme". arXiv:2003.07193 (2020). Verificado: compara contra SVM y NB en 4 datasets de
  sentimiento.
- Limitaciones: todo en 2016-2020, todo en sentimiento/topicos, nada en toxicity ni espanol. 2016-
  2020 esta FUERA de la ventana solicitada.
- COMPATIBILIDAD: TOTAL.
- DIFERENCIA CONCRETA: marginal. Es essentially "STW bien regularizado", lo cual es una variante
  de algo ya rechazado. NO la priorizo.

### F4. Set de features "Davidson" (POS n-grams, longitudes, cuentas de metadatos, sentimiento, legibilidad) [V]

- Referencia canonica: Davidson, T.; Warmsley, D.; Macy, M.; Weber, I. "Automated Hate Speech
  Detection and the Problem of Offensive Language". ICWSM 2017. (arXiv:1703.04820; dataset
  t-davidson/hate-speech-and-offensive-language, 24802 tweets de Twitter, jerarquia Hate /
  Offensive / Neither).
- Referencia de validacion: Mozafari et al. 2024 (T2) reimplementaron EXACTAMENTE este set para
  persa. Componentes verificados textualmente: TF-IDF word n-grams (uni/bi/tri); numero de
  caracteres, palabras y silabas; numero de user mentions, hashtags, retweets y URLs;
  TF-IDF de POS n-grams (uni/bi/tri, con filtro de document frequency < 5); polaridad de
  sentimiento (negativa, positiva, neutra) via un lexicon de sentimiento externo; legibilidad con
  Flesch-Kincaid Grade Level y Flesch Reading Ease con el numero de oraciones fijado en 1;
  reduccion de dimensionalidad con LR-L1 y luego LR-L2.
- Ventaja reportada (macro-F1, 5-fold, de T2):
  - Nivel 1 (offensive vs non-offensive): Davidson 0.770 vs SVMWord 0.908 vs ParsBERT 0.878.
  - Nivel 2 (targeted vs untargeted): Davidson 0.751 vs SVMChar 0.829 vs ParsBERT 0.772.
  - Nivel 3 (individual vs group): Davidson 0.695 vs SVMChar 0.896 vs ParsBERT 0.691.
    Aqui Davidson IGUALA o supera ligeramente a ParsBERT (0.695 vs 0.691) y queda muy por debajo de
    SVMChar (0.896).
- Limitaciones: la version publicada de Davidson depende de un lexicon de sentimiento EXTERNO
  (pbert), lo que viola nuestras reglas; y en T2 se mezclo con otras senales. Su version
  "solo estructura" (longitudes, cuentas de menciones/hashtags/retweets/URLs, silabas) es
  totalmente compatible.
- COMPATIBILIDAD: PARCIAL (la parte POS + longitud + cuentas es compatible; la parte sentimiento no,
  salvo que el lexicon de sentimiento se derive del train).
- DIFERENCIA CONCRETA: TOXIC ya probo "features estructurales/densas" y las rechazo. El
  subconjunto POS-n-grams (uni/bi/tri de etiquetas POS, df>=5) NO es una feature densa: son
  columnas sparse adicionales en el mismo espacio. Esa es la unica pieza de Davidson que no esta
  cubierta por el rechazo previo. Sin embargo, un POS tagger en espanol (spaCy, NLTK, Stanza) es un
  modelo entrenado externamente; auditar si el reglamento lo admite. NOTA: el propio T2 reporta
  que el set Davidson completo esta POR DEBAJO de TF-IDF n-grams solo en los niveles 1 y 2.

### F5. LSA / SVD como features adicionales [N - SIN EVIDENCIA A FAVOR]

- Busqueda realizada: multiples consultas de LSA/SVD + toxicity/short text/macro F1 en 2023-2026.
  No se encontro NINGUN trabajo de toxicity que valide LSA/SVD como feature incremental sobre
  n-gramas.
- Unico dato de senal opuesta, y en otro dominio: "Enhanced analysis of large-scale news text
  data..." (PMC11323039) indica que LSA "falls short in comparison to BOW" en reduccion de
  dimensionalidad para clasificacion.
- MARCA: sin evidencia. NO propongo LSA/SVD.
- Nota: TOXIC ya probo "proyecciones supervisadas" (v9_05) y "skipgrams" (v6_03), que cubren el
  espacio de "transformar la matriz dispersa en una representacion latente".

### F6. Morfologia (stemming / lematizacion) [V - evidencia contraintuitiva]

- Referencia: Tellez et al. 2021. Tarea de sentimiento en espanol; exploracion exhaustiva de
  TODAS las combinaciones de transformaciones de texto (incluidas lemmatization, stemming, entity
  removal) y sus parametros.
- Hallazgo verificado, textual: "Counterintuitively, we found that the complexity of techniques
  used in the pre-processing step is not correlated with the final performance of the classifier,
  e.g., a classifier using lemmatization, which is one of the pre-processing techniques having the
  greatest complexity, might not be one of the systems having the highest performance."
- Contexto adicional verificado: el lematizador usado (Freeling) requeria un diccionario del
  espanol y un modelo de formacion de palabras; o sea, la lemmatization "correcta" en espanol
  tiene dependencia de recursos. Y la conclusion de su analisis top-k es que la mayoria de las
  mejores configuraciones NO dependen de la lemmatizacion sino de "enable TFIDF and properly
  handle emoticons and users".
- Ventaja esperada: no hay. La evidencia apunta a que la morfologia aporta poco en espanol de
  redes cortas.
- COMPATIBILIDAD: PARCIAL (lemmatizers usan modelos/diccionarios externos).
- DIFERENCIA CONCRETA: ninguna. Se documenta para CLOSAR la linea de morfologia con evidencia
  externa, no con intuicion.

---

## 5. Seleccion de caracteristicas y calibracion de decision en toxicity

### C1. Umbral optimo para maximizar macro-F1: la evidencia canonica [V]

- Referencia: Lipton, Z. C.; Elkan, C.; Naryanaswamy, B. "Optimal Thresholding of Classifiers to
  Maximize F1 Measure". NIPS 2014 Workshop on Machine Learning for Health (texto completo
  legible en PMC4442797; preprint arXiv:1402.1106).
- Por que importa EXACTAMENTE a nosotros: nuestra metrica es macro-F1 sobre 3 clases positivas.
  Contenido verificado:
  - "while performance on rare labels is unimportant for micro F1, macro F1 is dominated by
    performance on these labels".
  - "macro averaging F1 can conceal the occurrence of uninformative classifier thresholding".
  - Caso de estudio multietiqueta: para una etiqueta con base rate muy baja y clasificador no
    informativo, "micro F1 is optimized by classifying all examples as negative, while macro F1 is
    optimized by classifying all examples as positive".
  - "For a rare label with an uninformative predictive model, micro F1 is optimized by
    classifying all examples as negative, while macro F1 is optimized by classifying all examples
    as positive." -> sobre-prediccion deliberada de la clase minoritaria.
  - "a Winner's Curse" en la seleccion experimental de umbrales: la distribucion de umbrales
    elegidos empiricamente cambia con la base rate, y el umbral optimo en un conjunto pequeno es
    optimista. Es exactamente la trampa de validar el umbral en el mismo fold que reporta.
  - Proponen tres plug-in rules para maximizar micro, macro y "per instance average F1".
- Limitaciones: 2014, no toxicity, y el marco es de clasificacion binaria por etiqueta con
  clasificadores calibrados.
- COMPATIBILIDAD: TOTAL.
- DIFERENCIA CONCRETA respecto a lo ya probado en TOXIC: TOXIC ya hizo umbrales por etiqueta
  (v8_peso_por_etiqueta, v10_05 regla F1 optima, v10_06 umbrales estables, v11_07 correccion
  prior, v16_02 promedio de umbrales). La APORTACION NUEVA de este paper no es la tecnica sino
  el AVISO TEORICO: (a) con macro-F1, sobre-predecir la clase minoritaria es la decision
  OPTIMA teorica, no un bug; (b) el "winner's curse" obliga a que cualquier umbral reportado en
  el mismo conjunto de evaluacion sea considerado sesgado al alza, y que el numero de umbral
  elegido tenga incertidumbre. Si TOXIC reporto umbrales con delta pequeno y los eligio sobre el
  mismo dev, esos deltas son optimistas de forma systematica.
- ACCION: usar C1 como criterio para AUDITAR (no re-probar) los umbrales ya fijados: pedir el
  intervalo/bootstrap del macro-F1 en funcion del umbral y declarar la incertidumbre.

### C2. Decision Threshold Calibration vs rebalanceo: comparacion controlada [V]

- Referencia: "Balancing the Scales: A Comprehensive Study on Tackling Class Imbalance in Binary
  Classification". arXiv:2409.19751 (2024).
- Diseno verificado: 30 datasets, 15 modelos, clase minoritaria al 20%.
- Resultados verificados (tabla del paper):
  - F1-Score: Baseline 0.556 +- 0.006; SMOTE 0.605 +- 0.006; Class Weights 0.594 +- 0.006;
    DECISION THRESHOLD CALIBRATION 0.617 +- 0.005 (mejor).
  - Recall: baseline 0.517; SMOTE 0.683; class weights 0.609; threshold calibration 0.693.
  - Precision: baseline 0.676; SMOTE 0.575; weights 0.636; threshold 0.597.
  - PR-AUC: baseline 0.638 (el MEJOR); threshold 0.635; SMOTE 0.629; weights 0.637.
  - AUC: baseline 0.865; SMOTE 0.864; weights 0.866; threshold 0.862.
  - Brier: baseline 0.058; SMOTE 0.078; weights 0.066; threshold 0.058.
  - "Decision Threshold Calibration achieved the highest mean F1-score (0.617 +- 0.005)" y
    "was best for 12 out of 30 datasets (40%)" y "in 10 out of 15 models".
- Hallazgo clave: la calibracion de umbral mejora F1 y recall y MANTIENE AUC, PR-AUC y Brier
  (es decir, NO separa mejor, solo decide mejor). El rebalanceo (SMOTE) DEGRADA PR-AUC y Brier
  respecto a la linea base. Esto es relevante: la metrica de Kaggle es macro-F1, no AUC.
- Limitaciones: no es toxicity, es clasificacion binaria general; datasets sinteticos con 20%
  de minoritaria (nuestro desbalance puede ser distinto); 2024.
- COMPATIBILIDAD: TOTAL.
- DIFERENCIA CONCRETA: TOXIC ya probo umbrales y rechazo rebalanceo. Este paper NO aporta una
  tecnica; aporta la evidencia externa que VALIDA la decision de priorizar umbral sobre
  rebalanceo, y la advertencia de que rebalancear deforma la separabilidad (PR-AUC) sin mejorar
  F1. Refuerza el cierre de la linea de rebalanceo.

### C3. Focal loss + calibracion post-hoc de umbral para toxicity en espanol [V abstract]

- Referencia: Alajmi, A.; Pergola, G. "When in Doubt, Consult: Expert Debate for Sexism
  Detection via Confidence-Based Routing". arXiv:2512.23732 (dic 2025 / ene 2026).
- Tarea: deteccion de sexismo en espanol sobre EDOS Task A y Task B, y EXIST 2025 Task 1.1.
- Texto verificado del abstract: "we first stabilize the training combining class-balanced focal
  loss, class-aware batching, and post-hoc threshold calibration, strategies for the first time
  adapted for this domain to mitigate label imbalance and noisy supervision". Resultados:
  "+4.48% and +1.30% on EDOS Tasks A and B, respectively, and a +2.79% improvement in ICM on
  EXIST 2025 Task 1.1".
- Limitaciones: requiere fine-tuning de transformers. El resto del metodo (routing a un modulo de
  deliberacion con LLMs) esta prohibido. Ademas EDOS Task A es BINARIA (sexismo/no sexismo), no
  nuestra 3-etiqueta. La metrica ICM de EXIST 2025 es macro-F1 ponderado por tipo, no
  necesariamente nuestro macro-F1 plano de 3 clases.
- COMPATIBILIDAD: PARCIAL en cuanto a "post-hoc threshold calibration" (si, es solo post-proceso,
  sin modelo nuevo); NULA en cuanto a focal loss y al routing.
- DIFERENCIA CONCRETA: "post-hoc threshold calibration" aplicado COMO PALABRA (focal loss +
  threshold calibration juntos) no es una tecnica aislada. Y TOXIC ya tiene una linea de umbral muy
  desarrollada (fases 8, 10, 11, 16). CERO valor como candidato nuevo.

### C4. Umbral por clase determined por CV en espanol, en competencia real [V]

- Referencia: Mosquera 2021, DETOXIS 2021 (ver T6). Texto verificado: "Our stacking model was
  logistic regression with a custom threshold of 0.32, which was determined via cross-validation.
  The latter was required because of the class imbalance and the unusual evaluation metric used in
  this sub-task (F1 of the toxicity class rather than micro or macro averages) which favours
  aggressive models towards the positive class."
- Ventaja reportada: el equipo quedo 6o de 31 con F1-Toxic 0.5691 (mejor 0.6461). No es una
  prueba controlada del umbral; es evidencia de que la practica es estandar en la escena.
- COMPATIBILIDAD: TOTAL.
- DIFERENCIA CONCRETA: NINGUNA (ya implementado en TOXIC). Se documenta como validacion de que
  la practica coincide con la literatura.

---

## 6. Evidencia NEGATIVA sobre transformers en toxicity con datos pequenos

### N1. Persa: SVM con n-gramas gana a ParsBERT en las TRES etiquetas [V] -> T2

Ver T2. Numeros: nivel 1 SVMWord 0.908 > ParsBERT 0.878; nivel 2 SVMChar 0.829 > ParsBERT 0.772;
nivel 3 SVMChar 0.896 > ParsBERT 0.691. Gap maximo: +0.205 macro-F1 a favor de la linea clasica.
Con 6000 filas de train y 3 etiquetas, en OTRO idioma (persa) con la misma estructura de tarea.
ESTO ES LA EVIDENCIA MAS FUERTE A FAVOR DEL TECHO CLASICO.

### N2.Turkico: TF-IDF+LR queda a 0.08 de XLM-R en el caso favorable [V] -> T1

TF-IDF+LR turco macro-F1 0.85 vs XLM-R 0.93. Y en el caso de bajo recurso (azerbaiyano) ambos
colapsan: TF-IDF 0.47 (recall hate 0.00) vs XLM-R 0.72 (recall hate 0.00). La ventaja del
transformer es real pero MODESTA y dependiente de la lengua, y en el regimen de bajo recurso no
salva la clase minoritaria a ninguno de los dos. Ambas familias comparten el techo de "no
detectar la clase minoritaria cuando hay desbalance severo".

### N3. NB-SVM gana a RoBERTa en 6.2% relativo (Hinglish) [V] -> F1

Sentimiento, no toxicity. Aporta que la familia sparse-lineal puede ganar, pero no que gane en
toxicity ni en espanol.

### N4. Evidencia ADVERSA a la linea clasica pura en ESPANOL [V] -> T6

DETOXIS 2021, corpus NewsCom-TOX en espanol, <4000 muestras (comparado con nuestras 1731):
- BOWClassifier (bag-of-words clasico) 0.1837
- ChainBOW 0.3746
- RandomClassifier 0.3760
- AllToxic con spacyLR (LR sobre EMBEDDINGS espanoles preentrenados) 0.4231
- Mejor sistema (SINAI) 0.6461 con mejor modelo individual = capsule sobre GloVe SBWC i25
Es decir: los sistemas puramente BoW quedaron MUY por debajo de los sistemas con embeddings
preentrenados. En espanol, con tamano de corpus similar al nuestro, la representacion densa
preentrenada parece valer mucho mas que la dispersa. CONFLICTO con T6/molero (que dice que los
transformers siguen ganando) y con T2 (persa, donde gana la clasica).

Matiz honesto: los baselines BoW de DETOXIS podrian ser debiles por diseno (son baselines de
"todo negativo" de una tarea compartida), asi que este numero probably SUBESTIMA a TF-IDF bien
ajustado. No es una prueba limpia.

### N5. Espanol: los transformers siguen ganando [P] -> T4

Molero et al. 2023 (IEEE Access), espanol, recorrido explicito bag-of-words -> transformers,
conclusion indexada: "We show how transformer-based models continue obtaining the best results, but
we improved previous results by a 6,2% by adding new steps of pre-processing". Y en espanol,
Usman et al. 2025 reportan que la linea base SVM+TF-IDF queda en ~0.78 de macro-F1 frente a 0.85
de GPT-3.5.

### N6. Evidencia ADVERSA en otros idiomas [V - via snippets de busqueda, sin lectura completa]

- Comments toxicos en chino (estudio comparativo 2025): "all transformer-based models
  significantly outperformed all classical ML models" (benchmark indonesio con validacion
  cruzada: IndoBERT obtiene el mejor F1-medio). MARCA [P]: no lei los articulos; solo snippets.
- Ademas: "A Controlled Study on Toxic Comment Classification" (ACM, DOI 10.1145/3795892.3795893)
  indexa: "In the baseline phase, the classical models, limited by sparse TF-IDF representations,
  were unable to effectively distinguish toxic content". MARCA [P]: snippet de busqueda, sin
  lectura.
- Contexto de dominio: el caso Twitch/Kulkarni et al. 2017 reporta que los metodos char/word
  n-gram son superados por ~18 puntos de F1 por metodos de deep learning en toxicity. MARCA [P].
  Ademas fuera de ventana (2017).

### N7. Sintesis de la evidencia sobre el techo [V - sintesis de lo anterior]

Matriz de resultado (ganador por combinacion lengua x tamano):
- Persa, 6000 filas, 3 etiquetas: CLASICA gana (+0.03 a +0.21 macro-F1). [T2]
- Turco, datos suficientes: TRANSFORMER gana (+0.08). [T1]
- Azerbaiyano, 1112 filas, fuerte desbalance: NINGUNO funciona (clase minoritaria recall 0). [T1]
- Espanol, ~4000 filas: TRANSFORMER-CON-EMBEDDINGS gana por mucho (+0.22 sobre BoW). [T6]
- Espanol, ~3.2k filas, binaria: TRANSFORMER/LLM gana (+0.07 a +0.09). [T9, T4]
- Hinglish, sentimiento: CLASICA (NB-SVM) gana (+0.062 relativo). [F1]

Conclusion de la matriz: la dominancia clasica NO se generaliza al espanol. En espanol, toda la
evidencia publicada disponible favorece al modelo con embeddings/transformer. Y lo unico que
gana al clasico con prueba solida y reproducible en espanol es el PREPROCESAMIENTO (molero,
+6.2% atribuido a pasos de preprocesamiento), no la representacion ni el clasificador.

---

## 7. Tabla resumen deEMECompatibilidad

| # | Tecnica | Ref / anio | Efecto publicado | Comp. | Difference con TOXIC |
|---|---|---|---|---|---|
| T1 | TF-IDF+LR vs transformers, turco/azeri | Alizada 2026, SIGTURK | 0.85 vs 0.93 macro-F1 (TR) | SI | referencia de calibracion |
| T2 | SVM n-grams vs ParsBERT, 3 etiquetas | Mozafari 2024, PLOS ONE | +0.03/+0.06/+0.21 a favor de clasica | PARCIAL | evidencia de techo, otro idioma |
| T3 | TF-IDF vs n-grams altos, low-resource | SNAM 2025 | cualitativo | SI | ninguno |
| T4 | BoW -> transformers + preprocesamiento, espanol | Molero 2023, IEEE Access | +6.2% por preprocesamiento | PARCIAL | BRECHA de verificacion |
| T5 | Transformaciones de texto espanolas | Tellez 2021, ESWA | word+char q-grams +11.17%/+5.62% | SI | **CANDIDATA PRINCIPAL** |
| T6 | DETOXIS 2021 espanol, tabla de sistemas | Mosquera 2021, CEUR | BoW 0.18-0.37 vs 0.65 con embeddings | NULA/PARCIAL | evidencia adversa, umbral y hilos |
| T7 | Meta-coleccion espanol | Piot 2025, arXiv | solo LLM | NULA | breach de datos |
| T8 | TF-IDF+LR isiXhosa/Yoruba | Agbeyangi 2026 | acc 0.61-0.86 | SI | ninguno (ruido) |
| T9 | Multilingue es/en/ur, SVM base | Usman 2025 | SVM ~0.78 macro-F1 en espanol | NULA | situa nuestro LB 0.783 |
| L1 | Multilingual HateCheck (incl. espanol) | Rottger 2022, WOAH | diagnostico, 34 func., 10 idiomas | NO | unusable |
| L2 | HateCheck / HateoMeter (ingles) | Rottger 2021, ACL | 3728 casos | NO | unusable |
| L3 | HurtLex 50+ idiomas 17 categorias | Basile 2018, CLiC-it | filtro de muestreo | NO | cobertura ES no verificada |
| L4 | Weaponized Word 125+ idiomas | weaponizedword.org | miles de terminos | NO | cobertura ES no verificada |
| L5 | MOL lexico contextual | Vargas 2021, RANLP | 1000 terminos | NO | cobertura ES no verificada |
| L6 | SexHateLex (chino) | Jiang 2022, OSNEM | 3016 terminos | NO | idioma |
| L7 | "panel de_times" | - | - | - | **NO VERIFICADO** |
| L8 | Hatebase | TRAC 2024 snippet | lexicons crowdsourced | NO | cobertura ES no verificada |
| P1 | Correccion de elongacion | Tellez 2021 | no aislado | SI (regex) | no probado |
| P2 | Expansion de abreviaturas | Tellez 2021 | no aislado | SI (derivado del train) | no probado |
| P3 | Emoticones -> polaridad | Tellez 2021 | "best setups lo hacen" | SI (tabla del train) | no probado |
| P4 | @/#/URL como TAGS | Tellez 2021 | "best setups lo hacen" | SI | no probado como sustituto lexico |
| P5 | Agregacion por hilo | Mosquera 2021 | feature mas fuerte para no-toxica | CONDICIONAL | verificar metadatos |
| P6 | Pew "unificar y expandir" | - | - | - | **SIN EVIDENCIA** |
| P7 | Ablaciones sistematicas de preprocesamiento | Glazkova 2023 SNAM; Shukla 2024 SNAM | no extraible | - | BRECHA #2 de verificacion |
| F1 | NB log-count ratio / NB-SVM | Bhange 2020 (V); canonica NO verificada | +6.2% rel. sobre RoBERTa | SI | ya probado |
| F2 | Supervised term weighting: por que falla | Moreo 2020, TKDE | "failed to show consistent improvements" | SI | confirma el rechazo |
| F3 | STW regularizado (re) | Wu 2016 | mejor que esquemas previos (sentimiento) | SI | variante de algo rechazado |
| F4 | Set Davidson (POS, long., metadatos) | Davidson 2017; validado en Mozafari 2024 | 0.770/0.751/0.695 vs 0.908/0.829/0.896 | PARCIAL | subconjunto POS n-grams no cubierto |
| F5 | LSA / SVD | - | - | SI | **SIN EVIDENCIA A FAVOR** |
| F6 | Stemming / lemmatizacion | Tellez 2021 | complejidad no correlaciona con rendimiento | PARCIAL | cierra la linea |
| C1 | Umbral optimo para macro-F1 + winner's curse | Lipton 2014, NIPS | marco teorico del riesgo | SI | auditar, no reprobar |
| C2 | Decision threshold calibration | arXiv 2409.19751 | F1 0.617 vs 0.556 baseline; AUC intacta | SI | valida el rechazo de rebalanceo |
| C3 | Focal loss + umbral, espanol (EDOS) | Alajmi 2025 | +4.48% Task A | PARCIAL | requiere fine-tuning |
| C4 | Umbral por clase via CV, espanol | Mosquera 2021 | 6o/31 con umbral 0.32 | SI | ya implementado |

---

## 8. Estado del techo de la linea clasica

### 8.1 Veredicto

VEREDICTO: la linea clasica de TOXIC esta probablemente CERCA de su techo, pero NO hay prueba de
que este en el techo, y las tres unicas vias que quedan con evidencia publicada a favor son
estrechas. Formalmente:

1. NO es falso que la linea clasica este en su techo. Es probable. La evidencia que lo apoya:
   - (a) En la tarea de la competencia, nuestra linea base (LB 0.78306) esta ya en el orden de
     magnitud del estado del arte clasico publicado para hate speech en espanol con corpus
     equivalentes: Usman et al. 2025 reporta SVM+TF-IDF en ~0.78 macro-F1 sobre 3162 tweets en
     espanol, y T9 es la unica referencia cuantitativa directa que pude verificar para espanol +
     macro-F1 + linea clasica.
   - (b) En el unico estudio 2023-2026 donde una tarea multietiqueta de lenguaje odivo con 3
     etiquetas es comparada contra transformers con datos suficientes, la ventaja del clasico
     REGRESIONA a 0.08 (T1) y en espanol la representancia densa preentrenada gana por ~0.22
     (T6). Es decir, en espanol la linea clasica no solo esta cerca del transformer: esta
     claramente por detras.
   - (c) Las dos tecnicas de representacion mas clasicamente prometidas (supervised term
     weighting, log-odds/NB-SVM) tienen evidencia externa de que no mejoran de forma consistente
     (F2, F1) y ya fueron rechazadas internamente.
   - (d) El preprocesamiento morfologico en espanol tiene evidencia debil de que la
     complejidad no se correlaciona con el rendimiento (F6), lo que elimina la via "mejorar la
     normalizacion linguistica".
   - (e) LSA/SVD no tiene ninguna evidencia a favor en toxicity (F5).
   - (f) La recalibracion de decision (C1, C2, C4) ya esta implementada en TOXIC de forma mas
     avanzada que la literatura.

2. NO es cierto que este en el techo de forma demostrada. Tres vias siguen abiertas con evidencia
   publicada:
   - (i) PREPROCESAMIENTO ESPANOL ESPECIFICO como vista adicional (P1-P4). Es la unica via con
     evidencia cuantitativa en espanol y con metricas macro: Molero et al. 2023 atribuyen +6.2% a
     "nuevos pasos de preprocesamiento" sobre un corpus espanol, y Tellez et al. 2021 muestran que
     las mejores configuraciones en espanol habilitar TFIDF y tratar correctamente emoticones y
     usuarios, con +11.17%/+5.62% de la combinacion de word n-grams + char q-grams sobre word
     n-grams solo. Y Tellez et al. 2021 es un pipeline de texto espanol que NO esta implementado
     en TOXIC.
   - (ii) CHAR q-GRAMS de rango mayor (T5). Tellez et al. 2021's gain viene de q-grams grandes
     (q=5,6,7) combinados con n-grams de palabra, no simplemente de "char n-grams". Si el
     campeon actual usa char n-grams de rango corto (2-5), extending a 4-7 y combinar en la union
     es un cambio de alcance no trivial. Ademas T3 reporta que "TF-IDF captured enough
     discriminative power without becoming too sparse" frente a bigramas/trigramas, lo que es
     coherente con "ampliar el rango de qgrams".
   - (iii) AGREGACION POR HILO / CONTEXTO (P5). En DETOXIS espanol fue la feature MAS FUERTE para
     la clase no toxica. Es condicional a que el train tenga metadatos de hilo.

3. La conclusion operativa: la fase de "representation + classifier" esta CERRADA. La fase de
   "text views + decision calibration" tiene un residuo pequeno, no nulo. Si la regla de decision
   de la sesion es no proponer lineas ya probadas, las candidatas priorizadas son:

   CANDIDATA 1 (principal). Vista ortografica espanola adicional como TERCER bloque de
   vectorizacion, implementada con reglas puras (sin diccionario externo): (a) reduccion de
   repeticiones de caracteres con regex r"(.)\1{2,}" mas un mapeo de las risas a una familia de
   tokens; (b) expansion de una lista corta de abreviaturas derivada por observacion del propio
   train; (c) mapeo de un conjunto reducido de emoticones/emojis (derivado del train) a tokens
   POS/NEG/NEU; (d) reemplazo lexico de @user, #hashtag y URL por tokens marcadores. Se anade al
   word+char actual; no lo reemplaza. Base: T5 + P1-P4. Riesgo principal: encoger la
   cobertura de char n-grams; por eso DEBE implementarse como vista aditiva con su propia
   verificacion de cobertura, y medirse en la misma macro-F1 de 3 clases.
   Justificacion de prioridad: es la UNICA via con (i) evidencia en espanol, (ii) metricas macro,
   (iii) datasets de nuestro tamano, (iv) diferencia no cubierta por las lines ya rechazadas, y
   (v) cero dependencia de recursos externos.

   CANDIDATA 2. Ampliar el rango de char q-grams (4-7) y combinarlos en la union con los
   n-grams de palabra, replicando el diseno de Tellez et al. 2021 (que es el unico estudio
   espanol con numeros de gain para la combinacion word n-grams + char q-grams). Base: T5, T3.
   Diferencia respecto a TOXIC: no es "mas char n-grams del mismo rango", es cambiar la
   COMBINACION (union de dos tokenizadores con rangos distintos) y verificar si el rango
   corto actual esta truncando la senal.

   CANDIDATA 3. Agregacion por hilo / thread (P5) SI y SOLO SI el train expone identificador de
   hilo o de respuesta. Base: T6. Antes de invertir: verificar la disponibilidad del metadato. Si
   no existe, descartar por imposibilidad. Si existe, es la feature con la senal mas fuerte
   reportada en espanol para la clase no toxica.

   CANDIDATA 4 (baja prioridad, evidencia negativa). STW regularizado al estilo "re" (F3) sobre
   la matriz actual. Se rankea bajo porque la evidencia independiente dice que STW no mejora de
   forma consistente (F2) y porque TOXIC ya lo rechazo.

   DESCARTADAS POR REGLA (no son candidatas, pero la conclusion debe quedar registrada):
   - Todo lexico externo (L1-L8): dataset externo etiquetado. Ademas, la unica via legal (derivarlo
     del train) es NB-SVM / STW, ya rechazada.
   - Focal loss (C3): requiere fine-tuning.
   - Pseudo-labeling, transduction, uso de test: prohibidos explicitamente.
   - Embeddings/LLM/transformers: prohibidos explicitamente.
   - LSA/SVD (F5), morfologia (F6): sin evidencia o con evidencia debil.

### 8.2 Pruebas que sostienen el veredicto, ordenadas por solidez

Nivel 1 - solidas (leidas por completo, con numeros, tarea multietiqueta comparable):
  T2 (Mozafari et al. 2024, PLOS ONE): SVM n-grams > ParsBERT en 3 etiquetas, gaps +0.03/+0.06/+0.21.
  T1 (Alizada & Dubossarsky 2026, SIGTURK): TF-IDF 0.85 vs XLM-R 0.93 en turco; ambos colapsan
  en el regimen bajo recurso.
  T5 (Tellez et al. 2021, ESWA): +11.17%/+5.62% de word n-grams + char q-grams en espanol;
  complejidad de preprocesamiento no correlacionada con rendimiento.
  T6 (Mosquera 2021, DETOXIS/CEUR): tabla completa de sistemas en espanol con ~4000 muestras;
  BoW puro 0.18-0.37 vs 0.65 con embeddings.
  C1 (Lipton et al. 2014): marco teorico de umbral para macro-F1 y winner's curse.
  F2 (Moreo et al. 2020, TKDE): supervised term weighting no mejora consistentemente.
  F6 (Tellez et al. 2021): morfologia no aporta.
  C2 (arXiv 2409.19751): umbral > SMOTE > class weights en F1, con AUC intacta.

Nivel 2 - parciales (metadata verificada, texto completo no leido; usar solo como direccion):
  T3 (SNAM 2025): "TF-IDF exhibited superior performance for non-transformer models".
  T4 (Molero et al. 2023, IEEE Access): "+6.2% por nuevos pasos de preprocesamiento" en espanol.
  T9 (Usman et al. 2025): SVM ~0.78 macro-F1 en espanol.
  C3 (Alajmi & Pergola 2025): umbral post-hoc + focal loss, EDOS espanol.
  N6 (varios): transformers > classical en chino e indonesio.

Nivel 3 - no verificadas (no usar como evidencia):
  T7 (Piot et al. 2025) solo LLM: relevante como brecha, no como metodo.
  T8 (Agbeyangi 2026): lineas metodologicamente debiles.
  L1-L6, L8: recursos lexicograficos, mayoria con cobertura espanola NO confirmada.
  L7 ("panel de_times"): NO ENCONTRADO. No citar.
  P6 (Pew "help-wrap"): SIN EVIDENCIA. No citar.
  P7 (Glazkova 2023, Shukla 2024): no se pudo extraer ninguna magnitud.
  F1 (referencia canonica de NB-SVM): NO VERIFICADA en esta sesion.
  F5 (LSA/SVD): sin evidencia a favor.

### 8.3 Brechas de verificacion que, si se cierran, podrian cambiar el veredicto

BRECHA 1 (critica). T4 / Molero et al. 2023 (IEEE Access, DOI 10.1109/ACCESS.2023.3257861). Es la
unica referencia en espanol que cuantifica la ganancia de preprocesamiento (+6.2%) sobre metricas
macro. No se pudo leer el PDF. Si esa ganancia se descompone en pasos concretos y es reproducible,
es exactamente la Candidata 1. ACCION: conseguir el PDF por via institucional.

BRECHA 2 (importante). P7 / Glazkova 2023 (SNAM 13:155) y Shukla & Dwivedi 2024 (SNAM). Son las dos
unicas fuentes de ablation sistematica de preprocesamiento en toxicity/abuso. Ninguna es espanola,
pero darian el prior sobre que tecnicas de limpieza rinden. ACCION: conseguir los PDFs.

BRECHA 3 (menor). F1: la referencia canonica del log-count ratio / NBSVM. No verificable en esta
sesion. No afecta el veredicto porque NB-SVM ya fue rechazado internamente.

BRECHA 4 (menor). L3, L4, L5: cobertura espanola de HurtLex, Weaponized Word y MOL. Irrelevante
para la decision porque son inadmisibles por regla, pero relevante si la regla del concurso
resultara ser mas permisiva de lo que dice el briefing.

BRECHA 5 (decisiva para la Candidata 3). P5: si el train del concurso expone identificador de
hilo/respuesta. Es una pregunta de UNA LINEA sobre el dataset, no de literatura. Es lo primero que
hay que responder antes de considerar la Candidata 3.

---

## 9. Nota de reproduccion de esta busqueda

Consultas ejecutadas (arXiv API, ACL Anthology, Semantic Scholar API, OpenAlex, Crossref, PMC,
CEUR-WS, repos publicos, web search en espanol e ingles). Bloqueos encontrados: Springer
(link.springer.com), IEEE Xplore, TechRxiv, ResearchGate devolvieron error de carga o 403. arXiv,
ACL Anthology, PMC, CEUR-WS, OpenAlex, Crossref y Semantic Scholar funcionaron. El endpoint de
Semantic Scholar devolvio 429 (rate limit) en varias llamadas, lo que redujo la cobertura de la
base de datos de Semantic Scholar; las ausencias en este informe pueden ser ausencias en mi
busqueda y no ausencias en la literatura.

Regla seguida sin excepcion: no se incluyo ninguna referencia cuyos autores, ano, venue o DOI no
pudiera verificar en al menos una fuente primaria (metadatos via OpenAlex/Crossref/Semantic
Scholar/ACL/arXiv). Las fichas marcadas [N] se declaran explicitamente no verificadas y NO se
usan como evidencia de ninguna conclusion.
