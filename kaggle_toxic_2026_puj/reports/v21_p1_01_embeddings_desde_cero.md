# v21 P1/01 - Embeddings de palabra entrenados desde cero y composicion de documento

Fecha: 2026-09-26.
Alcance: revision bibliografica (2023-2026 con clasicos canonicos) para la fase 1 de v21.
Restricciones asumidas como inviolables (no se propone nada fuera de ellas):
NO transformers, NO LLMs, NO embeddings preentrenados de ningun tipo
(ni word2vec-google, ni fasttext-wiki, ni glove-twitter, ni deberta,
ni sentence-transformers, ni LASER/MUSE). Solo embeddings que el equipo
entrene desde cero sobre corpus NO etiquetado. Tampoco datasets externos
etiquetados.
Contexto: deteccion multietiqueta de hate speech (HS), toxicity (TR) y
aggressive language (AG) en tweets en espanol. 1.731 filas train, 1.154 test.
Metrica: macro-F1 de las 3 clases positivas.

Advertencia de formato: este archivo es ASCII puro y esta en espanol sin
acentos, por requisito de entrega.

---

## 0. Lo que la fase v5 ya ha agotado (no repetir)

Antes de proponer nada hay que fijar el punto de partida, porque varias de las
preguntas de este encargo ya fueron respondidas empiricamente por el equipo.
Datos tomados de `reports/embedding_phase_closeout_v5.md`,
`reports/embedding_architecture_v5_1.md`, `reports/embedding_complementarity.md`,
`reports/embedding_fusion_v5.csv` y `DECISIONS.md` (D-030 a D-033).

| Medicion | Valor |
|---|---|
| V03 (TF-IDF word(1,2) + char_wb(3,5) + LR) | corrected_cv 0.777211, LB 0.78306 |
| Embedding denso solo (fastText 200d, Wikipedia 10M) | 0.652981 |
| Fusion hstack con bloque denso normalizado L2 | 0.781766, delta +0.004555, IC95 [-0.003161, +0.012409] |
| Mejor variante de la barrida (window=3) | 0.782959, delta +0.005748, IC95 [-0.002084, +0.013625] |
| Barrida de 14 configuraciones de fastText | fusion entre 0.7767 y 0.7830 (rango 0.0063), 0 adoptables |
| Corpus 5M / 10M / 20M / 50M | fusion 0.7776 / 0.7812-0.7818 / 0.7776 / 0.7803 |
| Cobertura por tipo con corpus web | 0.6839 -> 0.8413, sin cambio en la aportacion |
| Bloque denso estandarizado en vez de L2 | 0.690105, delta -0.087106 |
| Stacking anidado (meta-modelo) | 0.767774, delta -0.009437, IC95 completamente negativo |
| Pooling A (media) / C (media+max) / B (media TF-IDF) | 0.6530 / 0.6388 / 0.6348 |
| Suelo aleatorio a la misma dimension | 0.5404 |
| Correccion neta del embedding sobre 1.483 filas | +3 filas (baseline), +6 filas (WS-3) |
| Rebanadas con mejora / sin mejora | elongaciones +0.0360, urls +0.0209, hashtags -0.0001, sin_hate_explicito -0.0028 |

Cinco conclusiones que este informe respeta como datos duros:

1. El **pooling ponderado por TF-IDF ya fue probado y perdio** (0.6348 contra
   0.6530 de la media simple). No es una candidata nueva.
2. La **barrida de hiperparametros de fastText esta agotada**. Cualquier
   propuesta que solo mueva `dim`, `window`, `minCount`, `subwords` o
   `negative` es repetir lo ya rechazado.
3. El **registro y el tamano del corpus estan agotados** con corpus enciclopedico
   y con corpus web general (D-032, D-033). La cobertura subio 2.5x y la
   aportacion no se movio.
4. El **stacking generico esta cerrado** (quinta vez que falla en el proyecto).
5. La combinacion por **hstack L2 con bloque denso completo** ya esta medida: da
   +0.0046 con IC95 que toca el 0. Cualquier cosa que solo cambie como se unen
   dos bloques de escala distinta actua sobre un rango de 0.0063.

Por lo tanto, la pregunta competitiva ("que representacion densa clasica,
entrenada desde cero, anade senal INCREMENTAL que TF-IDF word(1,2)+char_wb(3,5)
no tiene") tiene que responderse cambiando un eje **ortogonal** al
hiperparametro. Los ejes que quedan son: (a) la geometria post-entrenamiento,
(b) el objetivo de entrenamiento, (c) el tipo de composicion de documento,
(d) el registro del corpus, (e) la dimension del bloque denso.

---

## 1. Fichas de tecnicas candidatas

### FICHA 1. Post-procesado del espacio: centrado + eliminacion de las D primeras componentes principales

- **Referencia principal**: Mu, J., Viswanath, P. (2018). *All-but-the-Top:
  Simple and Effective Postprocessing for Word Representations*. ICLR 2018.
  arXiv:1702.01417. DOI 10.48550/arXiv.1702.01417.
- **Referencia complementaria**: Arora, S., Liang, Y., Ma, T. (2017). *A Simple
  but Tough-to-Beat Baseline for Sentence Embeddings*. ICLR 2017.
  arXiv:1703.11175 (OpenReview id SyK00v5xx).
- **Referencia de respaldo con la misma mecanica pero sobre matrices de
  conteo**: Levy, O., Goldberg, Y., Dagan, I. (2015). *Improving Distributional
  Similarity with Lessons Learned from Word Embeddings*. TACL 3:211-225.
  DOI 10.1162/tacl_a_00134.
- **Anio**: 2015, 2017, 2018. (Clasico canonico del subcampo; no hay sustituto
  2023-2026 para esta operacion.)

- **Tarea**: mejora de representaciones de palabra y de sentence embeddings
  ya entrenadas, sin volver a entrenar. Intrinseca (similitud de palabra,
  analogia, categorizacion de conceptos) y **extrinseca** (similitud textual STS,
  clasificacion de texto).
- **Dataset / poblacion**: Mu & Viswanath evaluan word2vec y GloVe publicados,
  en ingles, en 7 conjuntos de similitud de palabra, 2 de categorizacion de
  conceptos, analogia, y **5 conjuntos downstream de clasificacion de texto con
  4 arquitecturas distintas**. Arora et al. evaluan STS 2012-2015, SICK y
 STS Benchmark, en ingles. Levy et al. evaluan word2vec vs modelos de conteo en
  ingles.
- **Representacion**: se resta la media global del vocabulario y se proyectan
  fuera las D primeras direcciones de PCA (D=3 para word2vec, D=2 para GloVe).
  En Arora et al. la version "SIF" = media ponderada por a/(a+p(w)) **mas**
  eliminacion de la primera componente. En Levy et al. la eliminacion de la
  primera componente se aplica a la matriz de cuentas.
- **Arquitectura**: no hay arquitectura. Es post-procesamiento lineal, O(|V| d).
  La variante de documento se puede aplicar directamente sobre la matriz de
  vectores de documento (n x d), no solo sobre el vocabulario.
- **Preprocesamiento**: ninguno adicional. Los hiperparametros son D y el
  criterio de estimacion del vector comun (todo el vocabulario en Mu &
  Viswanath; el conjunto de sentencias en Arora et al.).
- **Ventaja reportada (con numeros)**:
  - Mu & Viswanath: mejora **consistente** en los tres niveles lexicales; en
    downstream, la representacion procesada gana en **34 de 40** combinaciones
    (5 conjuntos x 4 arquitecturas) con una mejora media del **2.85 %**, y en
    las 6 restantes los dos resultados son comparables. Ninguna empeora.
  - Arora et al.: la ponderacion SIF mejora la similitud textual **entre un 10 %
    y un 30 %** frente a la media simple, y el resumen abstract del articulo
    afirma que supera a RNN y LSTM.
  - Numeros reproducidos por Pagliardini et al. (2018, Tabla 4): GloVe sin
    post-procesamiento da 0.685 en STS 2014 y 0.722 en SICK 2014; con SIF
    (W+R) sube a **0.735** y **0.729**. Es decir, la mayor parte de la ganancia
    de SIF (+0.050 en STS 2014) viene del bloque R (eliminacion de componente),
    no del bloque W (los pesos): sobre SICK el total es +0.007.
  - Levy et al. (2015): demuestran que **la mayor parte de la ventaja de
    word2vec sobre los modelos de conteo se explica por tres decisiones de
    diseno**, y la primera de ellas es eliminar la primera componente principal;
    las otras dos son ponderar correctamente las palabras raras y usar los
    vectores de palabra y de contexto a la vez. Ademas, esas tres correcciones
    se transfieren a los modelos de conteo y recuperan la mayor parte del
    hueco.
- **Limitaciones y cautelas**:
  - La critica publicada a SIF (arXiv:1909.13494, 2019, autores no verificados
    en esta sesion) apunta lo correcto para nuestro caso: **SIF no separa de la
    media cuando ambos usan eliminacion de componente principal**. Es decir, el
    efecto de la ponderacion por frecuencia se evapora al hacer R a los dos.
    Esto importa mucho: la parte que gana en la literatura es R, no W.
  - Mu & Viswanath eliminan solo las D primeras; el *whitening* completo
    (dividir por los valores singulares) es otra cosa y **no** esta respaldado
    por estas referencias. En downstream, aplicar whitening completo a un bloque
    de 200 dimensiones con 1.483 documentos de entrenamiento es exactamente el
    patron que ya produjo -0.0871 cuando el bloque denso se estandarizo
    (D-031 punto 4). Riesgo alto.
  - Todos los resultados publicados son en ingles y en oraciones/documentos
    medianos, no en tweets cortos de espanol.
- **Compatibilidad con nuestras reglas**: **SI**. No usa nada preentrenado:
    es una funcion determinista de los vectores que el equipo ya entreno. No
    necesita corpus adicional. Se puede aplicar al vocabulario o al bloque de
    documento. Debe ajustarse **dentro de cada fold de entrenamiento** (el
    vector comun y las direcciones de PCA se estiman con los datos del fold),
    nunca con test.
- **Diferencia concreta respecto a lo ya probado en TOXIC**: no cambia ningun
  hiperparametro de fastText, no cambia el corpus, no cambia el pooling, no
  cambia el clasificador y no cambia la combinacion. Cambia la **geometria de
  los vectores que ya se entrenaron**, eliminando las direcciones dominantes.
  Y hay una justificacion especifica para nuestro caso, no generica: el
 odgeTH principal de un embedding de palabra suele ser una direccion de
  frecuencia (los documentos y el vocabulario comparten esa direccion porque
  las palabras mas frecuentes dominan el coseno). Esa es precisamente la
  direccion que TF-IDF **ya** codifica de forma explicita. Eliminarla del bloque
  denso deja solo la parte del bloque denso que no es redundante con TF-IDF.
  Ademas, la evidencia local del propio equipo (WS-3 gana con ventana 3, la
  menor) dice que la senal util es **local**; es coherente con que el bloque
  denso esta dominado por componentes de baja frecuencia/escala y que la
  senal discriminante viva en las direcciones de menor varianza.

---

### FICHA 2. Embeddings con objetivo dirigido a la tarea (task-oriented)

- **Referencias verificadas**:
  - Liu, Q., Huang, H., Gao, Y., Wei, X., Tian, Y., Liu, L. (2018). *Task-oriented
    Word Embedding for Text Classification*. COLING 2018, pp. 2023-2032.
    DOI 10.18653/v1/C18-1172.
  - Zheng, N. et al. (2015). *Task-Oriented Learning of Word Embeddings for
    Semantic Relation Classification*. arXiv:1503.00095.
  - Tang, D., Wei, F., Yang, N., Zhou, M., Liu, T., Qin, B. (2014). *Learning
    Sentiment-Specific Word Embedding for Twitter Sentiment Classification*.
    ACL 2014, pp. 1555-1565. DOI 10.3115/v1/P14-1146.
  - Kenter, T., Borisov, A., de Rijke, M. (2016). *Siamese CBOW: Optimizing Word
    Embeddings for Sentence Representations*. ACL 2016, pp. 941-951.
  - Monnar, A. A., Perez Rojas, J., Poblete Labra, B. (2024). *Cross-lingual hate
    speech detection using domain-specific word embeddings*. PLOS ONE 19(7):
    e0306521. DOI 10.1371/journal.pone.0306521 (con fe de errata de 2025,
    DOI 10.1371/journal.pone.0323507).
- **Anio**: 2014, 2015, 2016, 2018, 2024.

- **Tarea**: los cuatro primeros cambian el objetivo de entrenamiento para que
  los vectores de palabra/organizacion sean discriminativos de una clase
  concreta (relacion semantica, sentimiento de Twitter, representacion de
  oracion). Monnar et al. hacen deteccion multilingue de habla de odio.
- **Dataset / poblacion**: Liu et al. usan **cinco** conjuntos de clasificacion
  de texto y declaran mejora significativa sobre el estado del arte. Tang et al.
  usan sentimiento en Twitter (ingles). Kenter et al. usanSTS/pafrasis.
  Monnar et al. usan tres corpora etiquetados de tweets de odio (ingles,
  espanol, italiano) en evaluacion monolingue y cross-lingual zero-shot.
- **Representacion**: word embeddings (o vectores de oracion) cuya geometria
  esta optimizada para separar clases, en lugar de para aproximar coocurrencia.
  En Monnar et al. la version concreta es word2vec 100d entrenado **desde cero**
  sobre un corpus de tweets recuperado con semillas lexicones de odio, es decir
  corpus no etiquetado pero de registro dominio-especifico.
- **Arquitectura**: skip-gram con negatives modificados (Task-oriented, SSWE) o
  CBOW siames para representacion de oracion (Kenter et al.). Monnar et al.:
  word2vec estandar 100d + alineamiento CCA.
- **Preprocesamiento**: en Monnar et al., las semillas provienen de Hurtlex, un
  lexico **etiquetado** de 50 Lenguas. Ese punto es un problema para nuestras
  reglas (ver compatibilidad). En Liu et al. y Tang et al. el corpus de entrada
  es no etiquetado y las etiquetas entran solo en la funcion de perdida.
- **Ventaja reportada (con numeros)**:
  - Liu et al. (2018): "significantly outperforms the state-of-the-art methods"
    en los cinco conjuntos. **No dan tabla en el resumen**: no he verificado las
    cifras exactas, asi que no las invento aqui.
  - Kenter et al. (2016): CBOW siames, entrenado para que la media de los
    vectores sea una buena representacion de oracion, frente a CBOW plano.
  - Monnar et al. (2024): los embeddings de odio **ganan a MUSE en todas las
    configuraciones monolingues** de los tres idiomas, quedan a **menos de 1
    punto de mBERT**, y son el mejor modelo en **3 de 6** configuraciones
    cross-lingual (y segundo en las otras 3), con un corpus de 10M de tweets en
    espanol. Eso es evidencia directa y reciente de que un embedding entrenado
    desde cero, sobre el registro correcto, aporta senal incremental en espanol
    y en la tarea de odio.
- **Limitaciones y cautelas**:
  - El riesgo clasico es sobreajuste: con 1.483 filas etiquetadas y 3 etiquetas,
    un objetivo que optimiza directamente la separabilidad puede memorizar. La
    mitigacion estructural es que los vectores se **comparten** entre las 3
    etiquetas y se entrenan sobre el corpus (millones de ocurrencias), no sobre
    1.483 filas.
  - Monnar et al. son **zero-shot cross-lingual**: su escenario de evaluacion no
    es el nuestro (nosotros si tenemos etiquetas de espanol). La parte que si
    se traspola es la de monolingue, y ahi el margen frente a una referencia
    fuerte esta entre "1 punto" y "gana a MUSE".
  - La comparacion de Monnar et al. es contra representaciones preentrenadas
    (MUSE, mBERT), no contra TF-IDF. **No hay en esa referencia evidencia de
    que su embedding de 100d supere a TF-IDF word+char como bloque de
    features**. Es un argumento a favor de "el registro del corpus importa", no
    un argumento a favor de "el bloque denso bate a TF-IDF".
  - ClassX (Kozil & Kaski, 2016), que suele citarse en este nicho, **no ha
    podido verificarse** en esta sesion (no aparece en Crossref ni en las
    busquedas). No la uso como referencia.
- **Compatibilidad con nuestras reglas**: **CONDICIONAL**, y se desdobla:
  - La variante *task-oriented con las etiquetas del propio train* (Liu, Zheng,
    Tang, Kenter) es **SI**: usa las etiquetas de la competencia, que son
    datos propios, no un dataset externo. No usa nada preentrenado.
  - La variante *semilla de lexico de odio externo* (Monnar et al. con Hurtlex)
    es **NO** tal cual: un lexico etiquetado externo es un recurso etiquetado
    externo. Habria que sustituir las semillas por el vocabulario del propio
    train, lo cual es un cambio de procedimiento y no una replica.
- **Diferencia concreta respecto a lo ya probado en TOXIC**: cambia el
  **objetivo de entrenamiento**, no los hiperparametros. La fase v5 entreno
  fastText con el objetivo estandar de coocurrencia (SGNS) y encontro que la
  calidad de la representacion mejoraba (0.6530 -> 0.6687) sin que eso se
  tradujera en aportacion. Aqui la hipotesis es distinta: un objetivo que ve
  las etiquetas puede producir vectores cuya documentacion ya esta alineada
  con HS/TR/AG, que es el unico tipo de senal que el analisis de rebanadas
  muestra que falta (hashtags -0.0001, sin_hate_explicito -0.0028: ahi el
  embedding actual no aporta nada).

---

### FICHA 3. Composicion por topic model sobre el vocabulario de embeddings (representacion de documento por topics)

- **Referencias verificadas**:
  - Das, R., Zaheer, M., Dyer, C. (2015). *Gaussian LDA for Topic Models with
    Word Embeddings*. ACL 2015, pp. 795-804. DOI 10.18653/v1/P15-1077.
  - Zhao, J., Mudgal, S., Liang, Y. (2018). *Generalizing Word Embeddings using Bag
    of Subwords*. EMNLP 2018. DOI 10.18653/v1/d18-1059.
  - Rui, W., Xing, K., Jia, Y. (2016). *BOWL: Bag of Word Clusters Text
    Representation Using Word Embeddings*. CCKS 2016. Springer LNCS.
    DOI 10.1007/978-3-319-47650-6_1.
- **Anio**: 2015, 2016, 2018. (Tecnica classica; no es de 2023-2026 pero sigue
  siendo la referencia canonica del mecanismo.)

- **Tarea**: construir una representacion de documento que **no** sea un pooling
  lineal de vectores de palabra, sino una distribucion sobre un conjunto de
  topics induceidos a partir del vocabulario de embeddings.
- **Dataset / poblacion**: Das et al. usan LREC, NYT, CCG y Science (evaluacion
  de calidad de topics, no de clasificacion). Zhao et al. usan varios
  benchmarks de similitud y de tareas downstream. Rui et al. usan clasificacion
  de documentos.
- **Representacion**: en vez de `mean(v_w)`, se calcula la proporcion de topic
  de cada documento (un vector de K numeros no negativos que suma 1). En Das
  et al. el topic se modela como una gaussiana sobre el **espacio de embeddings**
  en vez de una multinomial sobre tipos de palabra, lo que ademas permite
  asignar topic a palabras nuevas. En Zhao et al. la unidad de conteo es la
  **subpalabra** en lugar de la palabra. En Rui et al. la unidad es el **cluster
  de palabras** al que se asigna cada vector.
- **Arquitectura**: modelo generativo (LDA) entrenado con muestreo de Gibbs, o
  EM. Entrenamiento no supervisado sobre el corpus.
- **Preprocesamiento**: el mismo que para el embedding. En Das et al. la
  co-ocurrencia se define sobre las ventanas de la(oracion) y sobre las
  palabras del topic.
- **Ventaja reportada (con numeros)**: **no he verificado una tabla de
  clasificacion downstream con numeros comparables a los nuestros**. Lo que si
  puedo afirmar: Das et al. muestran que la version continua (gaussiana sobre
  embeddings) permite asignar topic a palabras no vistas, lo que un LDA
  multinomial sobre tipos no puede hacer; Zhao et al. muestran que contar
  subpalabras en lugar de palabras generaliza mejor la representacion. **No
  puedo dar un delta en F1**: marcamos esto como evidencia mecanica, no
  cuantitativa.
- **Limitaciones y cautelas**:
  - K topics es un hiperparametro nuevo con su propia rejilla. Con 1.483 filas
    de entrenamiento, K grande es varianza; K pequeno es informacion.
  - Es un modelo **generativo de bolsa de palabras**: pierde el orden y, en la
    version de Das et al., el propio paper reporta que **no** logra recuperar
    los "space topics" que el LDA categorico si encuentra.
  - Riesgo alto de solapamiento con TF-IDF: la proporcion de topics es una
    funcion determinista del recuento de palabras, que es precisamente lo que
    TF-IDF ya captura. El unico valor diferencial posible es que el eje de K
    dimensiones **mezcla** palabras por similitud distribucional en vez de
    tratar cada palabra como un eje propio. Eso reduce dimensionalidad, no
    anade informacion en un sentido estricto. Es el riesgo que hay que medir.
- **Compatibilidad con nuestras reglas**: **SI**. LDA se entrena desde cero
  sobre corpus no etiquetado, no usa embeddings preentrenados (el vocabulario
  de embeddings lo genera el equipo) y produce K features numericas que se
  pueden unir al bloque disperso. Debe ajustarse dentro del fold.
- **Diferencia concreta respecto a lo ya probado en TOXIC**: los tres poolings
  ya medidos (media, media+max, media ponderada por TF-IDF) son **todos** sumas
  lineales de vectores de palabra. No hay ninguno que cambie la unidad de
  conteo. Aqui la unidad de conteo pasa de "palabra" a "topic de cluster de
  palabras" (o subpalabra), de modo que la composicion es una **probabilidad
  sobre un conjunto latente** y no un promedio en el espacio original. Con K
  entre 20 y 50 el bloque es de dimension baja comparada con las 200 del
  embedding, lo que ademas lo protege del problema de escala que ya produjo
  -0.0871.

---

### FICHA 4. Sent2Vec: composicion aprendida end-to-end con n-gramas, desde cero

- **Referencia**: Pagliardini, M., Gupta, P., Jaggi, M. (2018). *Unsupervised
  Learning of Sentence Embeddings using Compositional n-Gram Features*
  (Sent2Vec). NAACL 2018, pp. 528-540. DOI 10.18653/v1/N18-1049.
  arXiv:1703.02507.
- **Anio**: 2018. (Es el trabajo de referencia de esta tecnica; no hay
  sucesor 2023-2026 que lo reemplace, y los 2023-2026 que existen usan
  transformers y estan prohibidos aqui.)

- **Tarea**: aprender una representacion de oracion/documento **no supervisada**,
  entrenando simultaneamente los vectores de palabra y la regla de composicion.
- **Dataset / poblacion**: Toronto Books (0.9B palabras), Wikipedia (1.7B),
  **Twitter (19.7B)**. Evaluacion downstream sobre MSRP, MR, CR, SUBJ, MPQA y
  TREC con regresion logistica. En el corpus de Toronto (**tweets**) es donde el
  rendimiento en tareas de clasificacion es peor (Tabla 3: supervised 82.0-82.2,
  unsupervised 65.9-66.7), lo cual es un dato de caucion importante para
  nuestra tarea.
- **Representacion**: `v_S = (1/|R(S)|) * sum_{w in R(S)} v_w`, donde R(S) es
  la lista de n-gramas de la oracion (unigramas y, opcionalmente, bigramas), y
  cada n-grama tiene **su propio vector fuente aprendido**. El vector de oracion
  es la media de esos vectores. La diferencia con el mean-pooling del equipo es
  que los vectores estan entrenados **para que esa media sirva**.
- **Arquitectura**: extension de CBOW a ventana variable = oracion completa.
  softmax aproximado por muestreo negativo. Coste O(1) operaciones vectoriales
  por palabra, tanto en entrenamiento como en inferencia. Hashing trick para
  los n-gramas de orden superior. Dropout sobre los n-gramas de cada oracion y
  regularizacion L1 sobre los vectores.
- **Preprocesamiento**: tokenizacion; el corpus se trata como coleccion no
  ordenada de oraciones. Se elimina el submuestreo de palabras de CBOW (que
  destruye parte de la sintactica y acorta la ventana efectiva) y se usan
  ventanas dinamicas, precisamente para que la composicion sea sobre toda la
  oracion.
- **Ventaja reportada (con numeros)**:
  - Macro promedio sobre tareas supervisadas y no supervisadas: **75.9** para
    uni+bi sobre Twitter (19.7B), contra **70.2** para CBOW y **68.5** para
    skip-gram (ambos sobre Toronto, 0.9B). Mejora de **+3.8 puntos** sobre el
    mejor no supervisado, segun los propios autores.
  - Tabla 1 (Toronto, downstream con LR): Sent2Vec uni+bi **82.0** de media
    sobre 5 tareas, frente a **79.1** de CBOW, **78.5** de skip-gram y **80.7**
    de un BoW **unigram TF-IDF**.
  - Tabla 4: contra GloVe + SIF (840B de palabras, 42 veces mas corpus que el
    de Twitter de Sent2Vec), Sent2Vec uni+bi gana en downstream supervisado
    (0.835 contra 0.807) aunque pierde algo en STS 2014 (0.701 contra 0.735).
  - La adicion de bigramas "no ayuda mucho" en las tareas no supervisadas pero
    da un empujon en las supervisadas (82.0 contra 81.4 en uni).
  - Un hallazgo de interpretacion util: el perfil de las **normas** de los
    vectores aprendidos sigue la hipotesis de Luhn (1958), es decir, el modelo
    **aprende solo** a downweightar los terminos muy frecuentes y tambien los
    muy raros, Leaving the mid-frequency terms as the discriminative ones. Es
    decir, el modelo aprende una ponderacion de documentos mejor que un
    esquema fijo, y mejor que la de SIF.
- **Limitaciones y cautelas**:
  - En el corpus de **tweets**, que es el registro que nos interesa, el
    rendimiento es **claramente peor** que en Wikipedia/libros. Asi que la
    evidencia a favor es mas debil de lo que sugiere la media de Toronto.
  - Es un modelo **no supervisado**: no ve las etiquetas. Su ventaja es
    composicional, no discriminativa.
  - El corpus de entrenamiento tiene que estar segmentado en oraciones. En
    Twitter eso exige tokenizacion y limpieza de URLs y menciones, que es
    justamente donde el analisis de rebanadas del equipo muestra que el
    embedding actual ya aporta (+0.0209 en urls, +0.0042 en menciones).
- **Compatibilidad con nuestras reglas**: **SI**. Entrena desde cero, sin
  supervision y sin nada preentrenado. La salida es un bloque denso de
  dimension reduccible (700 en el paper, pero la arquitectura admite
  truncamiento) que se une al bloque disperso.
- **Diferencia concreta respecto a lo ya probado en TOXIC**: el equipo probo
  **tres reglas de pooling** sobre vectores fijos. Aqui la regla de composicion
  **se entrena junto con los vectores**, y la funcion objetivo es
  "predecir las palabras de la oracion desde el promedio de sus vectores
  fuente". Es el unico caso de la lista donde la composicion no es un
  promedio post-hoc sino un parametro aprendido. Ademas introduce n-gramas
  **dentro del espacio denso**, que es informacion que el bloque de char
  n-grams de V03 ya tiene, asi que su mecanismo diferencial es
  probablemente la parte **distribucional** (la norma aprendida por palabra,
  con downweighting automatico de extremos), no los bigramas.

---

### FICHA 5. Objetivo de conteo explicito: PPMI + SVD (en vez de SGNS)

- **Referencias verificadas**:
  - Levy, O., Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix
    Factorization*. NIPS 2014, pp. 2177-2185.
  - Levy, O., Goldberg, Y., Dagan, I. (2015). *Improving Distributional
    Similarity with Lessons Learned from Word Embeddings*. TACL 3:211-225.
  - Deerwester, G., Dumais, S. T., Furnas, G. W., Landauer, T. K., Harshman, R.
    (1990). *Indexing by Latent Semantic Analysis*. JASIS 41(6):391-407.
- **Anio**: 1990, 2014, 2015. (La referencia canonica del objetivo; el
  argumento de por que SGNS es una aproximacion con peso sigue vigente.)

- **Tarea**: obtener vectores de palabra resolviendo **exactamente** la
  factorizacion de la matriz de PMI en vez de aproximarla con SGD.
- **Dataset / poblacion**: Levy y Goldberg usan un corpus de wikipedia en
  ingles con vocabulario de 189.533 para palabra y para contexto; evaluan
  similitud de palabra y analogia. Levy et al. 2015 usan el mismo tipo de
  corpus y comparan SVD, SGNS y GloVe a 500 dimensiones.
- **Representacion**: matriz palabra x contexto de PMI, se pasa a PPMI (se
  recortan los valores negativos) y se factoriza con SVD truncada. Los vectores
  son los vectores singulares derechos.
- **Arquitectura**: conteo + SVD. No hay SGD, no hay muestreo negativo, no hay
  ventana ni subpalabra. Es enteramente determinista dado el corpus y el
  `min_count`.
- **Preprocesamiento**: ventana, eleccion de que contexto se usa (ventana,
  dependencia, o parrafo), y `min_count`. Levy et al. muestran que tres
  correcciones importan mas que el algoritmo: eliminar la primera componente
  principal, ponderar las palabras raras correctamente, y usar palabra +
  contexto.
- **Ventaja reportada (con numeros)**:
  - Levy y Goldberg (2014): "using a sparse Shifted Positive PMI word-context
    matrix to represent words improves results on two word similarity tasks and
    one of two analogy tasks". Y: "When dense low-dimensional vectors are
    preferred, **exact factorization with SVD can achieve solutions that are at
    least as good as SGNS's solutions for word similarity tasks**. On analogy
    questions SGNS remains superior to SVD" (atribuido a la natureza ponderada
    de la factorizacion de SGNS).
  - Levy et al. (2015): la conclusion central es que **la mayor parte de la
    ventaja de los embeddings neuronales sobre los modelos de conteo NO viene
    del algoritmo**, sino de decisiones de diseno e hiperparametros, y que esas
    decisiones se transfieren a los modelos de conteo y recuperan la mayor parte
    del hueco.
- **Limitaciones y cautelas**:
  - La matriz palabra x contexto de un corpus de 10M de tokens tiene del orden
    de 10^5 x 10^5 entradas. Es viable, pero el coste de memoria y de SVD
    truncada es sensible al `min_count` y a la ventana, que es un
    hiperparametro que el equipo **ya barrio** en fastText. La justificacion
    aqui no es "otro valor de ventana" sino "otro estimador".
  - SVD **no tiene subpalabras**: pierde la unica ventaja que el equipo si
  midio como real (+0.0138 de fastText sobre word2vec). Habria que combinarlo,
    lo cual ya no es PPMI-SVD puro.
  - La ventaja de SVD sobre SGNS en estos articulos es en **similitud de
    palabra**, no en clasificacion downstream. No hay en estas referencias
    evidencia de que PPMI-SVD supere a fastText en un clasificador.
- **Compatibilidad con nuestras reglas**: **SI**. Es el metodo de conteo
  canonico, se entrena desde cero, no usa nada preentrenado ni etiquetado.
- **Diferencia concreta respecto a lo ya probado en TOXIC**: el equipo si
  entreno un control **word2vec** (sin subpalabras) y si lo midio
  (embedding_downstream_results.csv), pero **no** una PPMI-SVD. La barrida de
  hiperparametros fue sobre el espacio de fastText. Aqui cambian el estimador
  (exacto en vez de SGD aproximado) y la estructura (V de V'' en vez de matriz
  de coocurrencia). Ademas ofrece acceso directo a una descomposicion espectral
  que interactua con la Ficha 1 de forma natural.

---

### FICHA 6. Subpalabras: estado del arte y por que no es candidata nueva

- **Referencias verificadas**:
  - Bojanowski, P., Grave, E., Joulin, A., Mikolov, T. (2017). *Enriching Word
    Vectors with Subword Information*. TACL 5:135-146.
    DOI 10.1162/tacl_a_00051. arXiv:1607.04606.
  - Wieting, J., Bansal, M., Gimpel, K., Livescu, K. (2016). *Charagram: Embedding
    Words and Sentences via Character n-grams*. EMNLP 2016, pp. 1504-1515.
    DOI 10.18653/v1/d16-1157.
- **Anio**: 2016, 2017.

- **Tarea**: cobertura de palabras fuera de vocabulario (OOV) y de palabras
  raras mediante unidades de subpalabra.
- **Dataset / poblacion**: Bojanowski et al. entrenan en dumps de Wikipedia de
  **nueve idiomas, entre ellos espanol**, y evaluan en 10 conjuntos de
  similitud (incluye WS353 espanol) y 4 de analogia.
- **Representacion**: cada palabra es una bolsa de n-gramas de caracter con
  limites `<` y `>`; el vector de la palabra es la **suma** de los vectores de
  sus n-gramas, con hashing FNV-1a a 2 x 10^6 cubos.
- **Arquitectura**: skip-gram con negativos, con la funcion de puntuacion
  sustituida por la suma sobre n-gramas. 1.5 veces mas lento que skip-gram.
- **Preprocesamiento**: n-gramas de 3 a 6 caracteres, umbral de rechazo 10^-4
  para submuetrear, `min_count` 5.
- **Ventaja reportada (con numeros)**:
  - Cobertura OOV: la version `sisg` (OOV representado por suma de n-gramas) es
    **siempre al menos tan buena como** `sisg-` (OOV con vector nulo), en los
    10 conjuntos. Ejemplos concretos: ingles WS353 71 -> 71, ingles RW 43 -> 47,
    ruso HJ 59 -> 66, rumano WS353 48 -> 54, aleman Gur350 61 -> 70.
  - **Espanol es el idioma donde el efecto es mas pequeno.** En WS353 espanol:
    skip-gram 57, CBOW 58, sisg- 58, sisg 59. Los autores lo dicen
    explicitamente: "the effect of using character n-grams is more important for
    Arabic, German and Russian than for English, French or Spanish".
  - Modelado de lenguaje (perplejidad, 5 idiomas): LSTM sin preentrenar /
    skip-gram / sisg = espanol **157 / 150 / 145** (3 % de mejora sobre skip-gram),
    Checo 339/339/312, ruso 262/237/206. Mejora mayor en lenguas eslavas.
  - Datos (seccion 5.4), que es la cifra mas relevante para nuestro caso:
    en Gur350, `sisg` entrenado con el **5 %** de la wikipedia alemana (66)
    supera a `cbow` entrenado con el **100 %** (62); en RW, `sisg` con el
    **1 %** de la wikipedia inglesa (45) supera a `cbow` con el 100 % (43).
    Es decir, **los embeddings con subpalabra necesitan mucho menos corpus**.
    Los autores concluyen ademas que su modelo **se satura rapido** y que anadir
    mas datos no siempre mejora.
  - Tamano de n-gramas (tabla 4): 3-6 es una eleccion razonable; la inclusion de
    n-gramas largos (n >= 6) es importante, y n = 2 es peor que n >= 3.
- **Limitaciones y cautelas**: la mejora en espanol es marginal (2 puntos de
  correlacion en el mejor caso). Es la lengua donde el modelo morphologico
  basedo en caracteres aporta menos, porque el espanol es bastante menos
  aglutinante que el checo o el ruso.
- **Compatibilidad**: **SI**, pero **ya esta hecho**. El baseline del equipo es
  fastText con `subwords` 3-6, y la barrida de subwords 5/6/7 ya salio
  toute. Charagram es una alternativa de *hashing* (hashes de n-gramas de
  caracter con 5-gramas, no 3-6) pero la unica palanca que cambia es el
  numero de n-gramas, que es exactamente la variable ya barrida.
- **Diferencia concreta respecto a lo ya probado**: **casi ninguna**. Charagram
  y fastText comparten el mecanismo (un vector por n-grama de caracter, palabra
  como suma/bag). Cambiar `subwords` por "5-gramas de caracter con hash" es la
  misma variable con otro nombre. **No es candidata.** La subpalabra ya esta
  pagada.

---

### FICHA 7. Registro de corpus dominio-especifico de la tarea

- **Referencia**: Monnar, A. A., Perez Rojas, J., Poblete Labra, B. (2024).
  *Cross-lingual hate speech detection using domain-specific word embeddings*.
  PLOS ONE 19(7): e0306521. DOI 10.1371/journal.pone.0306521. Fe de errata:
  PLOS ONE 20(4): e0323507, DOI 10.1371/journal.pone.0323507.
- **Anio**: 2024. (Es la referencia mas reciente y directamente relevante que
  he podido verificar.)

- **Tarea**: deteccion de habla de odio multilingue, con embeddings
  **entrenados desde cero** sobre corpus no etiquetado de tweets.
- **Dataset / poblacion**: 30M de tweets en ingles y **10M en espanol**,
  obtenidos de Twitter usando como semillas los terminos de un lexico publico
  de odio (Hurtlex). Evaluacion sobre tres corpora etiquetados de tweets de odio
  (ingles, espanol, italiano), en 9 configuraciones y 12 modelos.
- **Representacion**: word2vec **100d**, por idioma, entrenado desde cero sobre
  el corpus recovered con semillas; despues alineado con CCA entre idiomas.
  En nuestro caso monolingue la parte de CCA es irrelevante.
- **Arquitectura**: word2vec estandar.
- **Preprocesamiento**: el de word2vec. Lo importante es el **corpus**: tweets,
  no articulos.
- **Ventaja reportada (con numeros)**:
  - Gana a MUSE en **todas** las configuraciones monolingues.
  - Queda a **menos de 1 punto** de mBERT.
  - Es el **mejor** modelo en 3 de 6 configuraciones cross-lingual y segundo en
    las otras 3.
  - Los autoresargumentan que el corpus sesgado hacia odio hace que los vectores
    capturen "matices linguisticos especificos" y que las representaciones de uso
    general "contienen ruido de contextos no hostiles, lo cual degrada el
    rendimiento de los detectores de habla de odio".
  - Hallazgo cualitativo relevante: en el espacio de odio, el vecino mas cercano
    del italiano "migranti" es "illegals" en ingles y "palestinos" en espanol, no
    "migrants" / "migrantes". En MUSE el vecino es la traduccion literal. Es
    decir, el corpus de dominio **redefine** la vecindad hacia el significado de
    la tarea.
- **Limitaciones y cautelas**:
  - Requiere una descarga nueva y grande (10M de tweets). La decision de no
    escalar corpus en el proyecto (D-031, D-032, techo autorizado de 500M sin
    usar) fue una decision **tomada por evidencia**, no por coste, y esta ficha
    la reabre porque ofrece un mecanismo que los corpus ya probados no ofrecen:
    el **registro**, no el tamano. El corpus web mC4 que ya se probo es registro
    web generico, **no** registro de redes sociales.
  - Hurtlex es un lexico **etiquetado** externo (prohibido). Habria que
    reconstruir las semillas con el vocabulario del propio train.
  - El comparador de la referencia es MUSE/mBERT, no TF-IDF word+char. No hay
    en ese paper evidencia de que el bloque denso de 100d supere a TF-IDF como
    features.
- **Compatibilidad con nuestras reglas**: **CONDICIONAL**. El metodo es
  legal (texto no etiquetado, entrenamiento desde cero), pero (a) requiere
  autorizacion de gobierno de datos para una descarga nueva y (b) las semillas
  no pueden venir de un lexico etiquetado externo.
- **Diferencia concreta respecto a lo ya probado en TOXIC**: la fase v5 probo
  un corpus **web general** (mC4-es) y concluyo que el registro no era la
  palanca. Esa conclusion sigue siendo valida para mC4. Lo que Monnar et al.
  cambian es el **dominio semantico** del corpus (tweets, sesgados a odio), que
  es un eje distinto del registro generico. Es la unica ficha de este informe
  que ataca el problema desde los datos y no desde la representacion.

---

### FICHA 8. Union de bloque disperso + bloque denso: que dice la evidencia (y que no dice)

- **Referencias verificadas**:
  - Lui, M. (2012). *Feature Stacking for Sentence Classification in
    Evidence-Based Medicine*. ALTA 2012, pp. 134-138. (Taller; evidencia debil.)
  - Weinberger, K., Dasgupta, A., Langford, J., Smola, A., Attenberg, J. (2009).
    *Feature Hashing for Large Scale Multitask Learning*. ICML 2009, pp.
    1113-1120.
- **Referencia de apoyo sobre reduccion**: Deerwester et al. (1990) (LSA) y
  Kadhim, A. I., Cheah, Y.-N., Hieder, I. A., Ali, R. A. (2017). *Improving
  TF-IDF with Singular Value Decomposition (SVD) for Feature Extraction on
  Twitter*. IEC 2017 Proceedings Book. DOI 10.23918/iec2017.16. (Conferencia de
  ingenieria civil, sin revision por pares tematica; se marca como tal.)

- **Respuesta a la pregunta 3 (union vs concatenacion vs proyectar denso a
  disperso)**:
  - **SIN EVIDENCIA SOLIDA ENCONTRADA.** No he localizado ningun estudio
    publicado, con comparacion controlada, que enfrenten en clasificacion de
    texto (a) union simple de TF-IDF con un bloque denso de embeddings, contra
    (b) reducir el bloque denso con TruncatedSVD y luego unir, contra (c)
    reducir el bloque disperso con LSA y luego unir. La busqueda explicita
    devolvio comparativas de otras categorias (seleccion de rasgos,
    clasificadores) y no el contraste que se pide.
  - La unica evidencia directa located es Lui (2012), un trabajo de taller de
    5 paginas con un corpus clinico pequeno. Su observacion es util como aviso
    de riesgo, no como guia de diseno: "los conjuntos de rasgos son muy
    heterogeneos; algunos son grandes y dispersos (BOW) y otros pequenos y
    densos (rasgos estructurales). El peso relativo entre conjuntos de rasgos es
    dificil, y simplemente concatenar los vectores de rasgos a menudo llevaba a
    situaciones en las que **anadir mas rasgos reducia la precision global**". Su
    solucion fue el apilamiento por rasgos, que en el equipo **ya esta cerrado**
    (delta -0.0094 con IC95 completamente negativo, quinta vez que el stacking
    generico no aporta en este proyecto).
  - Sobre la **escala de los bloques**: la evidencia mas fuerte que tengo es
    **interna** y es concluyente. Estandarizar el bloque denso (column
    standardization) hace caer la fusion de 0.781766 a 0.690105, con los mismos
    datos. Normalizar L2 por documento llega a 0.781766. Es decir, el problema
    no es "que tecnica de union" sino "cual es la escala relativa". Eso ya esta
    resuelto por convencion (L2) y no por investigacion adicional.
  - Sobre **proyectar el bloque denso con TruncatedSVD antes de unir**: no hay
    evidencia publicada que lo respalde **para este proposito**. Y hay una
    razon tecnica para sospechar que **empeoraria**: reducir 200 dimensiones a,
    digamos, 20 por SVD es una proyeccion **no supervisada** que descarta
    precisamente las direcciones de menor varianza, y la evidencia del propio
    equipo dice que la senal util esta en las direcciones **locales y de baja
    escala** (gana con ventana 3, mejora en elongaciones y URLs). Es
    exactamente el perfil del error de SVD sobre un bloque de embeddings.
  - Lo que si tiene respaldo para "proyectar antes de unir" es
    **Weinberger et al. (2009)**, pero con otro alcance: muestran que un
    hash aleatorio de baja dimension del espacio de rasgos combinado conserva el
    rendimiento de un modelo lineal con L2 a una fraccion del tamano. Eso
    justifica **reducir la dimension total de la union por hashing**, no
    justificar TruncatedSVD sobre el bloque denso. No es lo que se pidio, pero es
    lo que la evidencia sostiene.
- **Compatibilidad**: SI en todas las variantes, pero el beneficio esperado es
  bajo y el riesgo de dano alto.
- **Diferencia respecto a lo ya probado**: minimo. El hstack L2 ya esta medido y
  el apilamiento ya esta cerrado. Cualquier reformulacion de la union esta
  operando sobre un rango de 0.0063.

---

### FICHA 9. Reduccion de dimensionalidad no transformacional sobre el bloque denso

- **Referencias**:
  - Deerwester, G., Dumais, S. T., Furnas, G. W., Landauer, T. K., Harshman, R.
    (1990). *Indexing by Latent Semantic Analysis*. JASIS 41(6):391-407.
    (LSA: la referencia canonica de SVD sobre una matriz dispersa de conteos.)
  - Kadhim, A. I., Cheah, Y.-N., Hieder, I. A., Ali, R. A. (2017). *Improving
    TF-IDF with Singular Value Decomposition (SVD) for Feature Extraction on
    Twitter*. IEC 2017. DOI 10.23918/iec2017.16.
- **Tarea**: reducir la dimension del bloque denso antes o despues de unirlo,
  para efecto en clasificacion lineal.
- **Ventaja reportada**: Kadhim et al. reportan mejora al aplicar SVD a TF-IDF
  para extraccion de rasgos en Twitter. **Calidad de la evidencia: baja.** Es
  una publicacion de conferencia de ingenieria civil, sin revision por pares
  tematica, y no he verificado su protocolo ni sus numeros.
- **Contraevidencia, y es la que aplica**: no la busco en la literatura porque
  la literatura no la hace falta. El argumento en contra es **propio y mecanico**:
  reducir 200 dimensiones a un numero pequeno por SVD es una proyeccion **no
  supervisada** que descarta precisamente las direcciones de **menor varianza**,
  y la evidencia interna del equipo dice que la senal util esta ahi (gana con
  ventana 3; mejora en elongaciones +0.0360 y URLs +0.0209). Truncar el bloque
  denso es, en terminos de la geometria que el equipo ya midio, tirar la senal
  que se quiere conservar. Ademas, un SVD **no supervisado** no puede saber
  que direcciones son utiles para HS/TR/AG, que es justamente el problema. No
  lo propongo como candidato; lo documento para cerrarlo.
- **NMF sobre el bloque denso**: **SIN EVIDENCIA ENCONTRADA** de que NMF sobre
  un bloque de embeddings de documentos ayude a un clasificador lineal en
  clasificacion de texto. No lo propongo.
- **Compatibilidad**: SI tecnicamente. **Diferencia respecto a lo ya probado**:
  no medida. Y el argumento mechanistic (que las direcciones de baja varianza
  son las utiles) va en contra.

---

## 2. Respuestas directas a las cinco preguntas del encargo

### 2.1 Embeddings desde cero: CBOW vs skip-gram, negativos vs jerarquico, subpalabras y cobertura OOV

- Mikolov, Chen, Corrado, Dean (2013a). *Efficient Estimation of Word
  Representations in Vector Space*. ICLR Workshop. arXiv:1301.3781. (CBOW y
  skip-gram.)
- Mikolov, Sutskever, Chen, Corrado, Dean (2013b). *Distributed Representations
  of Words and Phrases and their Compositionality*. NIPS 2013, pp. 3111-3119.
  (Muestreo negativo, jerarquico, submuetreado, vectores de frase.)
- Bojanowski et al. (2017), TACL 5:135-146 (subpalabras, ver Ficha 6).
- Wieting et al. (2016), EMNLP (Charagram).

**Respuesta**: el equipo ya tiene el estimador canonico (fastText = skip-gram con
negativos y subpalabras, que es la combinacion que la literatura marca como
dominante en espanol). La evidencia publicada relevante para nuestro caso es
doble y negativa para nuestra expectativa:
  (a) **subpalabras ya estan**: dan +0.0138 downstream frente al control word2vec
      en la medicion del equipo, y la cobertura OOV esta garantizada por
      construccion (todo tipo word-OOV obtiene vector). Pero la mejora en
      espanol es de las mas pequenas entre los nueve idiomas (WS353 es: 57 ->
      59), y las rebanadas donde el subpalabra ayuda (elongaciones, URLs,
      hashtags) son las que char_wb(3,5) de V03 **ya cubre**.
  (b) **negativos frente a jerarquico**: la unica afirmacion canonica verificable
      la hace el propio Mikolov et al. (2013b) en el resumen de su paper:
      "We also introduce Negative Sampling, a simplified variant of Noise
      Contrastive Estimation (NCE) that learns **more accurate vectors for
      frequent words** compared to the hierarchical softmax". Es decir, la
      ventaja del muestreo negativo esta acotada a las **palabras
      frecuentes** y es de calidad de vector, no de accuracy downstream. Y el
      beneficio practico es de **velocidad de entrenamiento** (el jerarquico es
      O(log |V|) por palabra en vez de O(|V|)), no de calidad. En espanol, donde
      el equipo ya usa negativos, no hay caso publicado para volver atras.
      No he encontrado ninguna fuente con numeros en espanol que compare ambos
      en un clasificador downstream. Ademas, la barrida del equipo de
      `negative` 5/10 no movio nada, y `negative` es el unico de los cinco
      parametros barriados que ademas controla el coste de entrenamiento.
  (c) **CBOW vs skip-gram**: Bojanowski et al. (2017, Tabla 1) reportan en
      espanol WS353: skip-gram **57**, CBOW **58**. La diferencia es de un
      punto, y a favor de CBOW. En los ocho idiomas donde evaluan ambos, los
      dos estan practicamente empatados (De Gur350 61 vs 62, En WS353 72 vs 73,
      Ru HJ 59 vs 60). Es decir, **la eleccion CBOW/skip-gram es
      estadisticamente irrelevante**, lo que es coherente con que la barrida
      del equipo no encontrara nada moviendo parametros del mismo estimador.
  Conclusion: este eje esta agotado. La unica forma de obtener algo nuevo por
  aqui seria cambiar el **estimador** (Ficha 5, PPMI+SVD) o el **objetivo**
  (Ficha 2), no los hiperparametros.

### 2.2 Composicion de documento: que pooling funciona mejor en textos cortos, y hay evidencia de que el weighted-mean por TF-IDF bate al mean

**Respuesta corta y honesta: NO hay evidencia de que el weighted-mean por TF-IDF
bate al mean puro, y hay evidencia en contra en nuestro propio setting.**

- El equipo **ya lo midio**: pooling A (media) 0.6530, pooling C (media+max)
  0.6388, pooling B (media ponderada por TF-IDF) 0.6348. La media gana por
  0.018 sobre la ponderada. La interpretacion del propio D-031: "ponderar por
  IDF concentra el vector en pocas palabras, que es justo lo que el averaging
  debia distribuir". En un tweet corto (~15-25 tokens) el problema es
  precisamente la escasez de tokens: downweighting las palabras medias deja
  todavia menos senal.
- **Evidencia publicada que apoya la media ponderada, pero con una advertencia
  critica**:
  - Arora et al. (2017) reportan que SIF (media ponderada + eliminacion de la
    primera componente) mejora la similitud textualSTS **entre 10 % y 30 %**.
  - Pero la reproduccion de Pagliardini et al. (2018, Tabla 4) muestra que
    GloVe va de 0.685 a 0.735 en STS 2014 (+0.050) y de 0.722 a 0.729 en SICK
    2014 (**+0.007**). La ganancia grande esta en un benchmark de similitud; en
    SICK, que se parece mas a una tarea de relacion semantica, la ganancia es
    minima.
  - Y la critica publicada (arXiv:1909.13494, 2019) encuentra exactamente que
    **la ventaja de SIF sobre la media se reduce cuando se aplica
    eliminacion de componente principal a ambos**. Es decir, la parte que
    funciona es **R, no W**.
  - Ademas, la propia Arora et al. afirma en su resumen que "simple average of
    the initial word embeddings does not work very well" en su modelo generico,
    pero ese resultado es sensible a que la media se calcule sobre vectores **ya
    post-procesados** o no.
  - Ethayarajh (2018), *Unsupervised Random Walk Sentence Embeddings* (RepL4NLP,
    DOI 10.18653/v1/W18-3012), presenta uSIF (SIF con eliminacion parcial de
    m componentes) y reporta que supera a SIF en los promedios de STS 2012-2015
    y SICK, mientras que en tareas de clasificacion (SST, SICK-R, SICK-E) sus
    numeros con ParaNMT son **inferiores o iguales** a los de SIF (82.8 vs
    80.5 en SST a favor del baseline AVG+WR). Es decir, **en clasificacion la
    evidencia a favor de la ponderacion es debil o nula**, y la evidencia a
    favor de la eliminacion de componentes es solida.
- **Sobre mean+max**: Shen et al. (2018), *Baseline Needs More Love: On Simple
  Word-Embedding-Based Models and Associated Pooling Mechanisms* (ACL 2018,
  DOI 10.18653/v1/P18-1041), muestran que los modelos sin parametros de pooling
  (SWEM) igualan o superan a RNN/CNN de embeddings en la mayoria de los casos
  considerados, y proponen max-pooling por interpretabilidad y pooling
  jerarquico. No he verificado su tabla de comparacion media vs media+max con
  numeros.
- **Texto corto especificamente**: Shen et al. (2018, tablas 1 y 2, del
  competition entry que citan) muestran, en NLI, que mean (71.3), sum (70.7),
  last (70.9) y max (70.6) estan dentro del intervalo de confianza al 95 %: "no
  podemos afirmar que alguno de estos metodos sea estadisticamente diferente de
  los otros". Ese paper es Reimers/Gurevych-side, pero la conclusion
  (diferencias de pooling estadisticamente insignificantes) es la que
  corresponde.

**Conclusion operativa de 2.2**: no reintentar B ni C. La palanca publicada
real es la **eliminacion de componentes principales** sobre la media, no el
cambio de pesos. Eso es la Ficha 1.

### 2.3 Union sparse+dense: ver Ficha 8

Resumen: sin evidencia comparativa publicada para el contraste pedido; la
evidencia interna dice que el factor decisivo es la escala relativa, ya fijada
por convencion a L2; y hay una razon mechanistic concreta por la que
proyectar el bloque denso con TruncatedSVD **empeoraria** el resultado en vez
de mejorarlo.

### 2.4 Reduccion no transformacional para denso+sparse: ver Ficha 9

Resumen: SVD/LSA tiene respaldo debil para el lado disperso (y con mala calidad
de fuente en el caso de Twitter); no hay respaldo para el lado denso; y el
argumento mecanico en contra (truncar un bloque no supervisado descarta las
direcciones de menor varianza, que es donde el equipo vio la senal util) no
depende de ninguna referencia externa. NMF sobre el bloque denso no tiene
ninguna evidencia encontrada.

### 2.5 Distribucion-small-data: cuanta palabra hace falta y que indice monitorizar

- **Referencias**:
  - Sahlgren, M., Lenci, A. (2016). *The Effects of Data Size and Frequency Range
    on Distributional Semantic Models*. EMNLP 2016, pp. 975-980.
    DOI 10.18653/v1/D16-1099.
  - Bojanowski et al. (2017), TACL (seccion 5.4, ya citada).
  - Wendlandt, L., Kummerfeld, J. K., Mihalcea, R. (2018). *Factors Influencing
    the Surprising Instability of Word Embeddings*. NAACL 2018, pp. 2092-2102.
    DOI 10.18653/v1/N18-1190.
  - Luhn, H. P. (1958). *The automatic creation of literature abstracts*. IBM J.
    Res. Dev. 2(2):159-165. (Hipotesis de significancia por rango de
    frecuencia; es el criterio que Sent2Vec aprende empiricamente.)
  - Keuleers, E., Stevens, M., Mandera, P., Brysbaert, M. (2015). *Word
    knowledge in the crowd: measuring vocabulary size and word prevalence in a
    massive online experiment*. Q. J. Exp. Psychol. 68(8):1665-1692.
    DOI 10.1080/17470218.2015.1022560.
  - Firth, J. R. (1957). *A synopsis of English usage*. (Origen clasico de la
    distincion entre "reconocible" y "ocurre". **No re-verificada en esta
    sesion**.)

- **Cuanta palabra hace falta**:
  - La respuesta corta y contraintuitiva: **mucha menos de la que el equipo
    asumio, si el modelo tiene subpalabras**. Bojanowski et al. muestran que
    `sisg` con el 1-5 % de un dump de Wikipedia iguala o supera a CBOW con el
    100 %. Los autores concluyen ademas que su modelo **se satura rapido**: anadir
    mas datos no siempre mejora.
  - La respuesta larga (Sahlgren & Lenci 2016): los tres factores que determinan
    el rendimiento de un modelo de semantica distribuida son el **numero de
    muestras**, el **rango de frecuencias** y la **complejidad del modelo**, y
    no se pueden compensar entre si. Un corpus pequeno pero con un rango de
    frecuencias estrecho no sirve. **No he extraido sus numeros de tabla**
    (el PDF no fue accesible en esta sesion), asi que no los cito.
  - Wendlandt et al. (2018) es el contrapeso necesario: "even relatively high
    frequency words (100-200 occurrences) are often unstable", y analizan el
    efecto de esa inestabilidad en tareas downstream. Es decir, hay una
    **inestabilidad intrinseca** a estos embeddings que no se arregla con mas
    corpus y que puede estar contaminando el bloque denso del equipo. Merece
    una ficha propia si se decide seguir por esta via.
  - **Conciliacion con la evidencia interna**: el equipo observo que pasar de
    10M a 50M tokens mejoro el embedding aislado (0.6530 -> 0.6687) pero **no**
    la fusion (+0.0046 -> +0.0031). Eso es exactamente lo que predice
    Sahlgren & Lenci: el numero de muestras fija la calidad de la
    representacion, y la calidad de la representacion **no** se traduce
    automaticamente en aportacion incremental. La leccion transferible ya esta
    escrita en el cierre de fase v5 y sigue siendo valida.

- **Que indice de cobertura monitorizar**:
  1. **Type coverage por token** (proporcion de tipos del corpus de evaluacion que
     tienen vector) y **por ocurrencia** (proporcion de tokens de evaluacion
     cubiertos). Es la metrica que el equipo ya midio (0.6839 -> 0.8413). Es
     **necesaria pero ya se demostro que no es la palanca**. Mantenerla, pero no
     usarla para decidir.
  2. **Cobertura ponderada por frecuencia de tarea**: en vez de cobertura de
     tipos, cobertura de los tokens que realmente caen en las rebanadas que
     hoy fallan (hashtags, insultos, colectivos). Es una metrica que el equipo
     **no** ha calculado y que si discrimina.
  3. **Type-token ratio y tamano efectivo de vocabulario** (Baayen 2001,
     "Word frequency distributions", Natural Language Engineering 7(4)) en lugar
     del conteo bruto de tipos, porque con corpus mas grande el TTR baja
     mecanicamente y el conteo bruto no dice nada.
  4. **Estabilidad a traves de semillas** (Wendlandt et al.): cosine medio entre
     el vector de un tipo y su vector cuando se reentrena con otra semilla. Es
     un indice de fiabilidad que no se ha monitorizado y que es el que explica
     la parte de ruido del bloque denso.
  5. **Ranking de frecuencia en la distribucion de la tarea**: la hipotesis de Luhn
     (terminos de frecuencia media son los mas discriminantes) sugiere
     monitorizar si el corpus aporta volumen en el rango medio. Este es
     precisamente el rango que Wikipedia NO tiene y que las redes sociales si.
  6. En espanol especificamente: **cobertura de subpalabra efectiva** (que
    fraccion de los 3-6-gramas de caracter de cada palabra de evaluacion tienen
    cubo de hashing asignado y no nulos). Es el indice que realmente explica la
    cobertura OOV, y es distinto de la cobertura de tipos.

---

## 3. Ranking de candidatas por probabilidad de aportar senal INCREMENTAL sobre TF-IDF

Criterio: probabilidad de superar el gate de adopcion del proyecto (+0.005 con
IC95 que no contiene el 0), dado que el embedding actual aporta +0.0046 con
IC95 tocando el 0, y dado que 14 configuraciones de fastText se movieron en un
rango de 0.0063.

### 1. Eliminacion de componentes principales (centrado + retirar las D primeras direcciones) sobre el bloque denso

Por que es la primera:
  - Es **gratis**: una funcion determinista de los `.vec` que ya estan
    entrenados. No requiereDecision de gobierno, no requiere descarga, no
    requiere corpus nuevo, y cuesta segundos.
  - Cambia un eje que la barrida **no** toco. La barrida movio el estimador
    (dentro de fastText). Esto cambia la **geometria** del estimador ya
    entrenado.
  - Tiene **dos** referencias independientes que convergen en el mismo
    mecanismo: Mu & Viswanath (2018) reportan mejora en 34 de 40 combinaciones
    downstream con media del 2.85 %, y Levy et al. (2015) demuestran que la
    eliminacion de la primera componente es **la primera de las tres** decisiones
    que explican casi toda la ventaja de los embeddings sobre los modelos de
    conteo. Dos grupos, dos-Year, dos corporos, el mismo hallazgo.
  - Encaja con la evidencia interna: la mejora del equipo se concentro en
    elongaciones (+0.0360) y URLs (+0.0209), y **cayo** en hashtags (-0.0001) y
    en el estrato sin odio explicito (-0.0028). El estrato que falla es el que
    depende de significado, no de forma. La primera componente principal de un
    bloque de embeddings es tipicamente una direccion de frecuencia/escala, que
    es justo lo que TF-IDF ya codifica; retirarla es la forma mas directa de
    quitarle al bloque denso la parte que solapa con TF-IDF.
  - **Riesgo principal y como se acota**: no aplicar *whitening* completo
    (dividir por valores singulares). Ya hay un precedente en el proyecto de lo
    que hace estandarizar el bloque denso: -0.0871. Probar D en {0, 1, 2, 3, 5}
    y parar en cuanto el bloque deje de comportarse como antes.
  - La justificacion para no usar la ponderacion SIF (peso a/(a+p(w))): el
    equipo ya midio que la ponderacion por IDF **empeora** en tweets cortos, y
    la critica publicada confirma que la ventaja de SIF desaparece cuando ambos
    metodos usan eliminacion de componente. Se aplica **R sin W**.

### 2. Embeddings con objetivo dirigido a la tarea

Por que es la segunda:
  - Es la unica ficha que ataca el problema que el analisis de rebanadas
    identifica como el realmente faltante: significado de la tarea (hashtags,
    identificacion de colectivo, negacion). Un objetivo que ve las etiquetas de
    HS/TR/AG puede producir vectores cuya composicion este alineada con esas
    tres etiquetas.
  - Es **legal** en la variante Liu et al. (2018) / Tang et al. (2014) /
    Zheng et al. (2015) / Kenter et al. (2016): usa las etiquetas del propio
    train, no un dataset externo.
  - Es ortogonal a todo lo barriado: no es un valor nuevo de `dim`, `window`,
    `minCount`, `subwords` o `negative`.
  - **Riesgo principal**: sobreajuste. Se acota con (a) compartir los vectores
    entre las 3 etiquetas, (b) entrenar sobre el corpus y no sobre las 1.483
    filas, (c) usar la misma cross-validacion corregida con umbrales congelados
    de V03 que ya esta montada.
  - **Evidencia de calibracion**: Monnar et al. (2024) muestran que un word2vec
    entrenado desde cero sobre 10M de tweets de dominio de odio queda a menos
    de 1 punto de mBERT y gana a MUSE en todas las configuraciones monolingues
    de espanol. No es un histico nuestro, pero es el mejor dato disponible de que
    en espanol y en esta tarea el registro del corpus y el objetivo importan mas
    que el tamano.

### 3. Composicion por topics (proporcion de topics de LDA/cluster) como bloque de baja dimension

Por que es la tercera:
  - Es la unica composicion de la lista que **no** es una suma lineal de vectores
    de palabra, que es exactamente la familia que el equipo ya agoto (media,
    media+max, media TF-IDF). Cambia la unidad de conteo de "palabra" a
    "topic" o "cluster de palabras".
  - La dimension (K entre 20 y 50) la protege del problema de escala que ya
    costo -0.0871.
  - **Riesgo principal, y es serio**: la proporcion de topics es una funcion
    determinista del recuento de palabras, que es lo que TF-IDF ya captura. La
    razon por la que podria anadir algo es que los K ejes **mezclan** palabras
    por similitud distribucional (si "perra" e "idiota" caen en el mismo topic,
    ese topic es un eje que ninguna palabra de TF-IDF produce). Esa es
    precisamente la senal incremental que buscamos, pero tambien es la razon por
    la que podria ser redundante.
  - **Honestidad sobre la evidencia**: **no he verificado ninguna tabla de
    clasificacion downstream con numeros comparables** para estas tecnicas. La
    evidencia que tengo es mecanica (poder asignar topic a palabras nuevas) y
    no cuantitativa. Se propone con la etiqueta "exploratorio" del propio
    proyecto, nunca como mejora.

### 4. Sent2Vec (composicion aprendida end-to-end con n-gramas)

Por que es la cuarta y no la primera:
  - Es el unico caso donde la composicion se **entrena** en vez de calcularse
    despues, y su hallazgo de que la norma de los vectores sigue la hipotesis de
    Luhn (los terminos medios son los discriminantes) es una senal de que
    aprende una ponderacion de documento mejor que cualquier esquema fijo.
  - Le baja el puesto la evidencia de que **en corpus de tweets** su rendimiento
    en tareas de clasificacion es visiblemente peor que en Wikipedia/libros
    (supervisado 82.0-82.2 frente a 83.3-83.5; no supervisado 65.9-66.7 frente
    a 66.2-68.3), y nuestro texto es de tweets.
  - Los bigramas no son la aportacion diferencial: el bloque de char n-grams de
    V03 ya cubre n-gramas.
  - Es la mas cara de las cuatro (los autores reportan 6.5 h con 30 hilos para
    19.7B de palabras; para 10M de tokens seria mucho mas rapida).

**Mencion aparte, no rankeada**: registro de corpus dominio-especifico
(Ficha 7, Monnar et al. 2024). No la rankeo porque depende de una decision de
gobierno de datos (descarga nueva de gran volumen, que D-031/D-032 ceraron
por evidencia) y porque las semillas que usa son un lexico etiquetado externo.
Si el equipo reabre el gobierno de datos, esta ficha sube al puesto 2 por
encima de la 2 actual, porque es la unica que ataca la variable de dominio y
tiene un resultado publicado reciente y positivo en espanol.

**Fuera del top 4 y con la evidencia en contra**: la union sparse+dense por
proyeccion (Ficha 8) y la reduccion no transformacional del bloque denso
(Ficha 9). No se propone ninguna. El razon es que el rango de la fusion ya esta
medido en 0.0063 entre 14 configuraciones, y no hay evidencia publicada de que
ninguna reformulacion de la union abra ese rango; ademas, la unica evidencia
comparativa directa que se encontro (Lui 2012) advierte que anadir rasgos
densos a un bloque disperso puede **reducir** la precision, y su solucion (el
apilamiento) ya esta cerrada con delta negativo en este proyecto.

---

## 4. Declaraciones explicitas de "sin evidencia"

Estas afirmaciones **no** tienen respaldo encontrado en esta sesion y no deben
presentarse como si lo tuvieran:

1. **Sin evidencia**: que el weighted-mean ponderado por TF-IDF bata al mean
   puro. Ademas hay evidencia en contra en nuestro propio setting (0.6348
   contra 0.6530) y la critica publicada a SIF indica que la ventaja de la
   ponderacion desaparece al aplicar eliminacion de componente a ambos metodos.
2. **Sin evidencia**: comparacion controlada publicada entre union simple,
   concatenacion y proyeccion con TruncatedSVD para texto con bloque disperso +
   bloque denso en un clasificador lineal. Solo se encontro Lui (2012), un
   trabajo de taller de 5 paginas, que advierte del riesgo pero no mide el
   contraste.
3. **Sin evidencia**: que LSA, SVD o NMF **sobre el bloque denso** de embeddings
   mejoren la clasificacion al unirlo con TF-IDF. La unica evidencia en
   direccion contraria es mecanica y propia: truncar el bloque denso descarta
   las direcciones de baja varianza, que son las unicas que el analisis de
   rebanadas identifica como senal incremental.
4. **Sin evidencia**: numeros de downstream comparables para las tecnicas de
   composicion por topic model (Gaussian LDA, bag of subwords, bag of word
   clusters). Solo evidencia mecanica.
5. **Sin evidencia**: que el jerarquico supere al muestreo negativo en
   clasificacion downstream en espanol. Lo unico que se puede citar es la
   afirmacion de los autores del muestreo negativo (Mikolov et al. 2013b), que
   acota su ventaja a "more accurate vectors for **frequent words** compared to
   the hierarchical softmax", es decir una afirmacion de **calidad de vector**
   para palabras frecuentes, no de accuracy de clasificacion. La ventaja
   practica del negativo es ademas de velocidad. No se localizo ninguna
   comparacion controlada de ambos en un clasificador en espanol.
6. **Sin verificar**: ClassX (Kozil & Kaski, 2016). No aparece en Crossref ni
   en ninguna busqueda realizada. No se usa.
7. **Sin verificar**: autores de la critica a SIF (arXiv:1909.13494, 2019). El
   contenido de la critica si esta verificado; los autores no.
8. **Sin extraer**: los numeros de tabla de Sahlgren & Lenci (2016). El PDF no
   fue accesible (error de content-type en el lector). Se cita la conclusion
   cualitativa, que esta confirmada por la referencia cruzada en multiples
   articulos, pero no las cifras.
9. **Sin verificar en linea en esta sesion**: Firth (1957). Se marca como origen
   clasico de la distincion cobertura, no como fuente de una cifra.
10. **Calidad de la fuente**, no ausencia de evidencia: Kadhim et al. (2017) es
    una publicacion de ingenieria civil sin revision tematica por pares; su
    resultado a favor de SVD sobre TF-IDF para Twitter no debe pesar mas que el
    argumento mecanico en contra (truncar un bloque no supervisado
    descarta las direcciones de baja varianza).

---

## 5. Referencias (todas verificadas en esta sesion salvo las marcadas)

1. Mikolov, T., Chen, K., Corrado, G., Dean, J. (2013a). Efficient Estimation of
   Word Representations in Vector Space. ICLR Workshop. arXiv:1301.3781.
   (CBOW y skip-gram.)
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., Dean, J. (2013b).
   Distributed Representations of Words and Phrases and their Compositionality.
   NIPS 2013, pp. 3111-3119. arXiv:1310.4546. (Muestreo negativo frente a
   jerarquico, submuetreado, vectores de frase.)
4. Levy, O., Goldberg, Y. (2014). Neural Word Embedding as Implicit Matrix
   Factorization. NIPS 2014, pp. 2177-2185.
5. Levy, O., Goldberg, Y., Dagan, I. (2015). Improving Distributional Similarity
   with Lessons Learned from Word Embeddings. TACL 3:211-225.
   DOI 10.1162/tacl_a_00134.
6. Bojanowski, P., Grave, E., Joulin, A., Mikolov, T. (2017). Enriching Word
   Vectors with Subword Information. TACL 5:135-146. DOI 10.1162/tacl_a_00051.
   arXiv:1607.04606.
7. Wieting, J., Bansal, M., Gimpel, K., Livescu, K. (2016). Charagram: Embedding
   Words and Sentences via Character n-grams. EMNLP 2016, pp. 1504-1515.
   DOI 10.18653/v1/d16-1157.
8. Arora, S., Liang, Y., Ma, T. (2017). A Simple but Tough-to-Beat Baseline for
   Sentence Embeddings. ICLR 2017. arXiv:1703.11175.
9. Mu, J., Viswanath, P. (2018). All-but-the-Top: Simple and Effective
   Postprocessing for Word Representations. ICLR 2018. arXiv:1702.01417.
   DOI 10.48550/arXiv.1702.01417.
10. Pagliardini, M., Gupta, P., Jaggi, M. (2018). Unsupervised Learning of
    Sentence Embeddings using Compositional n-Gram Features. NAACL 2018,
    pp. 528-540. DOI 10.18653/v1/N18-1049. arXiv:1703.02507.
11. Ethayarajh, K. (2018). Unsupervised Random Walk Sentence Embeddings: A
    Strong but Simple Baseline. RepL4NLP 2018. DOI 10.18653/v1/W18-3012.
12. Liu, Q., Huang, H., Gao, Y., Wei, X., Tian, Y., Liu, L. (2018).
    Task-oriented Word Embedding for Text Classification. COLING 2018,
    pp. 2023-2032. DOI 10.18653/v1/C18-1172.
13. Zheng, N. et al. (2015). Task-Oriented Learning of Word Embeddings for
    Semantic Relation Classification. arXiv:1503.00095.
14. Tang, D., Wei, F., Yang, N., Zhou, M., Liu, T., Qin, B. (2014). Learning
    Sentiment-Specific Word Embedding for Twitter Sentiment Classification.
    ACL 2014, pp. 1555-1565. DOI 10.3115/v1/P14-1146.
15. Kenter, T., Borisov, A., de Rijke, M. (2016). Siamese CBOW: Optimizing Word
    Embeddings for Sentence Representations. ACL 2016, pp. 941-951.
16. Das, R., Zaheer, M., Dyer, C. (2015). Gaussian LDA for Topic Models with
    Word Embeddings. ACL 2015, pp. 795-804. DOI 10.18653/v1/P15-1077.
17. Zhao, J., Mudgal, S., Liang, Y. (2018). Generalizing Word Embeddings using Bag
    of Subwords. EMNLP 2018. DOI 10.18653/v1/d18-1059.
18. Rui, W., Xing, K., Jia, Y. (2016). BOWL: Bag of Word Clusters Text
    Representation Using Word Embeddings. CCKS 2016.
    DOI 10.1007/978-3-319-47650-6_1.
19. Monnar, A. A., Perez Rojas, J., Poblete Labra, B. (2024). Cross-lingual hate
    speech detection using domain-specific word embeddings. PLOS ONE 19(7):
    e0306521. DOI 10.1371/journal.pone.0306521. Fe de errata: PLOS ONE 20(4):
    e0323507.
20. Chakraborty, R., Elhence, A., Arora, K., Mitkov, R. (2019). Sparse Victory:
    A Large Scale Systematic Comparison of count-based and prediction-based
    vectorizers for text classification. RANLP 2019. DOI 10.18653/v1/R19-1022.
21. Sahlgren, M., Lenci, A. (2016). The Effects of Data Size and Frequency Range
    on Distributional Semantic Models. EMNLP 2016, pp. 975-980.
    DOI 10.18653/v1/D16-1099.
22. Wendlandt, L., Kummerfeld, J. K., Mihalcea, R. (2018). Factors Influencing the
    Surprising Instability of Word Embeddings. NAACL 2018, pp. 2092-2102.
    DOI 10.18653/v1/N18-1190.
23. Deerwester, G., Dumais, S. T., Furnas, G. W., Landauer, T. K., Harshman, R.
    (1990). Indexing by Latent Semantic Analysis. JASIS 41(6):391-407.
24. Lui, M. (2012). Feature Stacking for Sentence Classification in
    Evidence-Based Medicine. ALTA 2012, pp. 134-138.
25. Weinberger, K., Dasgupta, A., Langford, J., Smola, A., Attenberg, J. (2009).
    Feature Hashing for Large Scale Multitask Learning. ICML 2009,
    pp. 1113-1120.
26. Kadhim, A. I., Cheah, Y.-N., Hieder, I. A., Ali, R. A. (2017). Improving
    TF-IDF with Singular Value Decomposition (SVD) for Feature Extraction on
    Twitter. IEC 2017. DOI 10.23918/iec2017.16. (Fuente de baja calidad.)
27. Keuleers, E., Stevens, M., Mandera, P., Brysbaert, M. (2015). Word knowledge
    in the crowd: measuring vocabulary size and word prevalence in a massive
    online experiment. Q. J. Exp. Psychol. 68(8):1665-1692.
    DOI 10.1080/17470218.2015.1022560.
28. Luhn, H. P. (1958). The automatic creation of literature abstracts. IBM J.
    Res. Dev. 2(2):159-165.
29. Shen, D., Wang, G., Wang, W., Min, M. R., Su, Q., Zhang, Y., Li, C., Henao,
    R., Carin, L. (2018). Baseline Needs More Love: On Simple Word-Embedding-
    Based Models and Associated Pooling Mechanisms. ACL 2018, pp. 440-450.
    DOI 10.18653/v1/P18-1041.
30. Eger, S., Rueckle, A., Gurevych, I. (2019). Pitfalls in the Evaluation of
    Sentence Embeddings. RepL4NLP 2019, pp. 55-60. DOI 10.18653/v1/W19-4308.
31. Critica a SIF. (2019). arXiv:1909.13494. (Autores no verificados.)

## 6. Nota sobre la cita 20 (Sparse Victory)

Se incluye porque es el contrapeso de mayor peso que he encontrado y conviene
que el equipo lo tenga presente antes de invertir mas esfuerzo en esta via:
sobre **73 conjuntos de datos publicos**, los vectorizadores dispersos
(tf-idf y feature hashing) superan a los modelos neuronales de embeddings de
palabra y de caracter (Word2Vec, GloVe, FastText, ELMo, Flair) en **61 de 73
conjuntos**, con un margen medio de **3 a 5 puntos de macro-F1**, y el resultado
es consistente en varias dimensiones de comparacion. Eso no dice que aqui no
pueda haber nada; dice que la **prior** de que un embedding denso anada
sobre un bloque disperso bien construido debe ser baja. El ranking de este
informe refleja esa prior: la primera candidata no pide un embedding mejor,
pide **quitarle al embedding que ya existe la parte que solapa con TF-IDF**.
