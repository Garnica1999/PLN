# HIPOTESIS DE TRABAJO - KAGGLE_TOXIC_2026_PUJ

- **Fecha**: 2026-09-22 (Fase 1, derivadas de `reports/estado_arte.md`).
- **Objetivo**: enunciar hipotesis falsables sobre que familias de features y que
  decisiones de modelado mejoran el F1 en las tres etiquetas (HS, TR, AG), cada
  una con evidencia de la literatura, un experimento que la probaria y la
  familia de features asociada.
- **Metrica de decision**: media column-wise de F1 binaria por etiqueta (primaria,
  D-009), con macro-F1 sobre 6 pares, micro-F1 y exact-match como secundarias.
  Toda comparacion se hace con la **misma** validacion
  (`StratifiedGroupKFold(k=5)`, grupos = near-duplicados, D-004) y con el mismo
  preprocesamiento, para que la diferencia sea atribuible a la hipotesis.
- **Regla de decision**: una hipotesis se acepta si mejora la metrica primaria en
  al menos **+0.005** absoluto en la media de los 5 folds y si la mejora es
  consistente en al menos 3 de los 5 folds. Si mejora en 1-2 folds o esta entre
  0.000 y 0.005, queda "no concluyente" y se documenta como tal. Si empeora, se
  rechaza y se registra el resultado negativo (los resultados negativos son
  entregables de la rubrica, no fracasos).

## Resumen y priorizacion

| Hip. | Enunciado abreviado | Familia de features | Riesgo | Impacto esperado | Orden |
|---|---|---|---|---|---|
| H1 | char n-grams 3-6 > word n-grams | Caracter | Bajo | Alto | 1 |
| H4 | word+char > cada una sola | Combinacion | Bajo | Alto | 2 |
| H8 | post-proceso jerarquico mejora F1 | Post-proceso | Muy bajo | Medio-alto | 3 |
| H9 | umbral por etiqueta != 0.5 mejora F1 | Post-proceso | Muy bajo | Medio-alto | 4 |
| H2 | lexico agresivo + 2a persona separa AG | Lexico + morfologia | Medio | Medio-alto | 5 |
| H3 | features de targeting separan TR | Sintaxis + targeting | Medio-alto | Medio | 6 |
| H5 | sentimiento/emocion complementa | Lexico afectivo | Medio | Medio | 7 |
| H6 | estilometria ayuda en textos cortos | Estilo de superficie | Medio | Medio | 8 |
| H10 | stacking > mejor modelo individual | Ensamble | Medio | Bajo-medio | 9 |
| H7 | lematizacion no aporta | Preprocesamiento | Bajo | Bajo (pero simplifica) | 10 |

Las hipotesis H1, H4, H8 y H9 son de **bajo riesgo y alto retorno**: se prueban
primero porque su coste computacional es minimo y su respaldo en la literatura es
el mas fuerte. H3 y H5 son las de mayor incertidumbre cientifica (son las que
mas aportan al notebook en terminos de narrativa) y por eso se prueban con
ablacion explicita y criterio pre-registrado.

---

## H1 - Los n-gramas de caracter 3-6 capturan evasion ortografica mejor que los n-gramas de palabras

- **Enunciado**: una representacion basada en n-gramas de caracter (rango 3-6,
  modo `char_wb`) obtiene mayor F1 que una representacion basada en n-gramas de
  palabras (1-2), manteniendo fijo el clasificador (SVM lineal) y la validacion.
- **Evidencia de la literatura**:
  - Nobata et al. (2016, `nobata2016abusive`, DOI 10.1145/2872427.2883062):
    los n-gramas de caracter estan entre las features mas informativas para
    lenguaje abusivo.
  - Stamatatos (2009, `stamatatos2009survey`, DOI 10.1002/asi.21001): en textos
    cortos y ruidosos los n-gramas de caracter son las features mas robustas.
  - Mishra et al. (2018, `mishra2018character`, DOI 10.18653/v1/W18-5101): la
    representacion a nivel de caracter entrenada desde cero rivaliza con la de
    palabras, lo cual es directamente aplicable bajo la prohibicion de
    embeddings preentrenados.
  - Argota Vega et al. (2019, `argotavega2019mineriaunam`, DOI
    10.18653/v1/S19-2079): el sistema que gano la Subtask A en espanol incluyo
    n-gramas de caracter entre sus features, sin redes neuronales.
  - Davidson et al. (2017, `davidson2017automated`, DOI
    10.1609/icwsm.v11i1.14955): uso n-gramas de caracter 2-4 como features
    centrales.
- **Razonamiento propio**: el corpus es ASCII sin diacriticos (D-010), lo que
  elimina parte de la variacion ortografica pero no la evasion por sustitucion
  de letras (`k` por `c`, `x` por `ch`), repeticion de vocales, separacion con
  puntos o simbolos, y concatenacion de palabras. Los n-gramas de caracter son
  la unica representacion que captura esos patrones sin necesidad de
  normalizacion explicita, y son inmunes a palabras fuera de vocabulario.
- **Experimento que la probaria**: **E01** `experimento_01_char_vs_word_ngrams.py`.
  Cuatro configuraciones con SVM lineal identico: (a) `char_wb` 3-6, (b) `char`
  2-5, (c) `word` 1-2 con `sublinear_tf`, (d) `word` 1-3. Se reportan las cuatro
  metricas y el tiempo de ajuste. Tambien se mide el solapamiento de vocabulario
  entre train y validacion para las dos representaciones, como diagnostico de
  por que gana una.
- **Familia de features**: n-gramas de caracter.
- **Criterio de falsacion**: si (a) no supera a (c) en la metrica primaria, H1
  se rechaza y la representacion principal pasa a ser word n-grams, lo que
  obliga a reconsiderar H7 y el diseno de E04.
- **Riesgo**: bajo. Es una comparacion directa con un coste de minutos.

---

## H2 - El lexico de agresividad combinado con marcas de segunda persona separa la etiqueta AG

- **Enunciado**: un bloque de features formado por (i) un lexico de agresividad y
  (ii) marcas de segunda persona (pronombres, posesivos, imperativos) mejora el
  F1 de AG sobre un modelo con solo n-gramas, y esa mejora es mayor para AG que
  para HS.
- **Evidencia de la literatura**:
  - Pereira-Kohatsu et al. (2019, `pereirakohatsu2019haternet`, DOI
    10.3390/s19214654): HaterNet usa pronombres de segunda persona y lexico
    agresivo como features centrales en espanol.
  - Wiegand et al. (2018, `wiegand2018inducing`, DOI 10.18653/v1/N18-1095):
    se puede **inducir** un lexico de palabras abusivas desde los propios datos
    usando POS y sintaxis, sin recurso externo. Resuelve la restriccion.
  - Bassignana et al. (2018, `bassignana2018hurtlex`, DOI
    10.4000/books.aaccademia.3085): HurtLex aporta un lexico de dano organizado
    en 17 categorias, con cobertura de espanol.
  - Argota Vega et al. (2019): incluyo "series de palabras agresivas" y patrones
    morfologicos (pronombre + verbo) entre las features ganadoras.
  - Davidson et al. (2017): documenta que la confusion entre "ofensivo" y "odio"
    es el error dominante; en nuestro corpus la frontera AG/HS es exactamente esa
    (26 casos con AG=1 y HS=0).
- **Razonamiento propio**: AG mide agresividad del autor, que es una propiedad
  del **acto de habla**, no del contenido tematico. Las marcas de segunda persona
  y los imperativos son indicadores directos de acto de habla dirigido; el lexico
  agresivo aporta la intensidad. HS, en cambio, depende del grupo objetivo.
- **Experimento que la probaria**: **E02** `experimento_02_lexico_agresividad.py`,
  con dos variantes:
  - **E02a (autocontenida, ruta preferente)**: lexico inducido de train mediante
    la metodologia de Wiegand et al. (2018): puntuar palabras por su asociacion
    con AG usando POS y frecuencia, con validacion interna para elegir el umbral.
  - **E02b (lexicon externo, D-007)**: HurtLex con las categorias relevantes,
    registrado previamente en `reports/lexicon_catalog.csv`.
  Ablacion de tres bloques: `base` (char+word) / `base+agresivo` /
  `base+agresivo+2a_persona`. Se reporta el F1 por etiqueta, no solo el agregado,
  para verificar que el efecto es mayor en AG.
- **Familia de features**: lexico de agresividad + morfologia de persona
  (POS/lema/morfologia de spaCy, D-003).
- **Criterio de falsacion**: si la mejora de AG es menor que la de HS, la
  hipotesis se rechaza: significaria que el lexico agresivo esta capturando odio
  y no agresividad, y habria que replantear la separacion HS/AG.
- **Riesgo**: medio. Depende de la calidad del lexico inducido y de la definicion
  real de AG en el corpus (que hay que caracterizar en Fase 2).

---

## H3 - Las features de targeting separan la etiqueta TR

- **Enunciado**: features que codifican la direccion del mensaje hacia una
  persona o grupo (menciones, hashtags, segunda persona, articulo determinado +
  sustantivo de grupo, tercera persona plural, vocativos) mejoran el F1 de TR
  sobre un modelo que no las incluye.
- **Evidencia de la literatura**:
  - ElSherief et al. (2018, `elsherief2018hatelingo`, DOI
    10.1609/icwsm.v12i1.15041): el odio se distingue por marcadores de
    "othering" (construccion de un "ellos" frente a un "nosotros"): primera
    persona plural, tercera persona y lenguaje de polaridad.
  - Argota Vega et al. (2019): construyo explicitamente "language patterns"
    (patrones morfologicos como pronombre + verbo y patrones lexicos) para
    distinguir si el tuit va dirigido a una persona o a un grupo. Ese fue su
    enfoque para la Subtask B.
  - Basile et al. (2019, `basile2019hateval`, DOI 10.18653/v1/S19-2007):
    la Subtask B define TR por el rango del target (grupo frente a individuo),
    una tarea estructuralmente distinta de la deteccion.
  - Zampieri et al. (2019, `zampieri2019predicting`, DOI 10.18653/v1/N19-1144):
    el target es una cabeza separada y mas dificil que la deteccion de ofensa.
  - Frenda et al. (2018, `frenda2018misogyny`, CEUR-WS Vol-2150): analisis
    linguistico de direccion hacia un grupo en espanol.
- **Razonamiento propio**: en el corpus, TR=1 solo ocurre con HS=1 (519/519).
  Esto significa que TR es, empiricamente, una **refinacion** de HS. La pregunta
  cientifica es si esa refinacion es aprendible con features de superficie o si
  depende de conocimiento del mundo (saber si "feminazi" apunta a un grupo).
  Las menciones `@usuario` y los vocativos son la senal mas directa de target
  individual; los articulos determinados con sustantivos colectivos
  ("los inmigrantes") son la senal de target grupal.
- **Experimento que la probaria**: **E03** `experimento_03_targeting_tr.py`.
  Bloque de features de targeting con seis sub-bloques ablacionables por
  separado: (1) menciones y hashtags, (2) pronombres de 2a persona,
  (3) pronombres de 3a persona plural, (4) articulo + sustantivo de grupo,
  (5) vocativos, (6) primera persona plural. Se mide el F1 de TR con y sin cada
  sub-bloque, y se hace un **analisis cualitativo de 100 ejemplos** de TR=1 y
  TR=0 para caracterizar la etiqueta (obligatorio por D-005 antes de optimizar).
- **Familia de features**: sintaxis y targeting (spaCy como anotador, D-003).
- **Criterio de falsacion**: si el bloque completo no mejora el F1 de TR, H3 se
  rechaza y se concluye que TR no es separable con features de superficie en este
  corpus; en ese caso TR se predice solo con la correlacion estructural
  (heredando de HS) y se documenta como limite.
- **Riesgo**: alto. La definicion de TR es ambigua (D-005) y esta es la
  hipotesis con mayor probabilidad de resultado no concluyente. Es tambien la que
  mas valor narrativo aporta al notebook.

---

## H4 - La union de n-gramas de palabra y de caracter supera a cada representacion por separado

- **Enunciado**: la representacion `word(1-2) + char_wb(3-6)` obtiene mayor F1
  que `word(1-2)` sola y que `char_wb(3-6)` sola.
- **Evidencia de la literatura**:
  - Argota Vega et al. (2019): la ganancia provino de un **framework
    combinatorio** que buscaba en el espacio de transformaciones; la
    configuracion ganadora combino varios tipos de n-gramas, no uno solo.
  - Garcia-Diaz et al. (2022, `garcia2022featurecombination`, DOI
    10.1007/s40747-022-00693-x): su contribucion central es evaluar
    **estrategias de combinacion** de features en espanol; las features
    linguisticas suman sobre TF-IDF.
  - Nobata et al. (2016): combino n-gramas de caracter con n-gramas de palabras,
    POS y lexicos.
  - **Contraevidencia obligatoria**: Nina-Alcocer (2019,
    `ninaalcocer2019haterecognizer`, DOI 10.18653/v1/S19-2072) encontro que las
    combinaciones de **tres** bloques degradaron la Subtask A, aunque cada bloque
    por separado mejorara. La combinacion no es monotona.
- **Razonamiento propio**: word n-grams capturan lexico tematico (grupos
  objetivo, insultos), char n-grams capturan forma y evasion. Son
  complementarios en principio, pero en un corpus de 1.731 filas el aumento de
  dimensionalidad puede dominar la senal si la regularizacion no es la adecuada.
- **Experimento que la probaria**: **E04** `experimento_04_union_word_char.py`.
  Barrido sistematico de 6 combinaciones con SVM lineal, regularizacion
  calibrada por validacion interna (`C` en `{0.1, 1, 10}`) y `min_df` en
  `{1, 2, 3}`. La clave metodologica es que **cada combinacion tiene su propio
  ajuste de regularizacion**: sin eso, el experimento mide el efecto de la
  dimensionalidad, no el de la complementariedad.
- **Familia de features**: combinacion de representaciones.
- **Criterio de falsacion**: si la union no supera al mejor individual en la
  metrica primaria, H4 se rechaza; se documenta junto con el hallazgo de
  Nina-Alcocer como evidencia convergente de que en corpus pequenos la
  combinacion de bloques no ayuda.
- **Riesgo**: bajo en coste, medio en resultado.

---

## H5 - Las features de sentimiento y emocion aportan informacion complementaria

- **Enunciado**: un bloque de features de polaridad y emocion mejora el F1,
  y su aporte se concentra en HS y AG (las etiquetas de actitud) mas que en TR.
- **Evidencia de la literatura**:
  - Nina-Alcocer (2019): `TFIDF+EMOTIONS` fue la mejor configuracion en EMR
    (0.711 frente a 0.629 del baseline, +0.082) en la Subtask B espanola. Es
    evidencia directa y con un metodo **totalmente compatible** con las
    restricciones de este proyecto.
  - Plaza-del-Arco et al. (2021, `plazadelarco2021multitask`, DOI
    10.1109/ACCESS.2021.3103697): modelar sentimiento y odio conjuntamente mejora
    la deteccion de odio en espanol. Es la evidencia mas fuerte de la
    complementariedad.
  - MacAvaney et al. (2019, `macavaney2019challenges`, DOI
    10.1371/journal.pone.0221152): la polaridad y la intensidad afectiva estan
    entre las senales que la literatura usa como puente entre opinion y odio.
- **Razonamiento propio**: la agresividad (AG) y el odio (HS) son actitudes
  afectivas; la polaridad negativa extrema y el enfado deberian correlacionar.
  Sin embargo, la literatura de odio implicito (ElSherief et al. 2021, DOI
  10.18653/v1/2021.emnlp-main.29) advierte que parte del odio es **frio y
  factual**, no afectivo, lo que acota el alcance esperado.
- **Experimento que la probaria**: **E05** `experimento_05_sentimiento_emocion.py`.
  Dos rutas: (a) lexicon autocontenido construido solo con train + WordNet/OMW
  (semillas de polaridad expandidas por synsets), y (b) lexicon externo
  documentado. Features: conteo y densidad por categoria afectiva, polaridad
  media, polaridad de la ultima oracion (el remate suele llevar la agresion).
- **Familia de features**: lexico afectivo y de emocion.
- **Criterio de falsacion**: si el bloque no mejora la metrica primaria o si la
  mejora se concentra en TR (donde no hay razon teorica para esperarla), la
  hipotesis se rechaza o se reformula.
- **Riesgo**: medio. La ruta autocontenida puede ser demasiado debil; por eso se
  mantienen ambas y se reportan por separado.

---

## H6 - La estilometria aporta en textos cortos

- **Enunciado**: un bloque de features estilometricas (n-gramas de palabras
  funcionales 2-4, n-gramas de puntuacion 2-4, medidas de forma: longitud,
  mayusculas, repeticion de caracteres, ratio de signos, ratio de hashtags y
  menciones) mejora el F1, con aporte concentrado en HS y AG.
- **Evidencia de la literatura**:
  - Stamatatos (2009, DOI 10.1002/asi.21001): en textos cortos y ruidosos, las
    **palabras funcionales** y los n-gramas de caracter son las features de
    estilo mas robustas.
  - Stamatatos (2017, `stamatatos2017distortion`, DOI 10.18653/v1/E17-1107):
    enmascarar el contenido tematico conservando la forma mejora la senal de
    estilo. Sugiere n-gramas de caracter sobre texto con palabras de contenido
    enmascaradas.
  - Argota Vega et al. (2019): incluyo explicitamente **n-gramas de palabras
    funcionales 2-4** (stopwords NLTK) y **n-gramas de signos de puntuacion 2-4**
    entre las features ganadoras, justificando los segundos como medida de
    "coherencia y cohesion" del texto escrito.
  - Nina-Alcocer (2019): `TFIDF+STRUCT` obtuvo el mejor F1 de la Subtask A
    (0.780 frente a 0.766 del baseline), con features estructurales de
    puntuacion, mayusculas y simbolos Unicode.
- **Razonamiento propio**: hay dos evidencias independientes (Argota Vega y
  Nina-Alcocer) de que las features de forma aportan en este corpus exacto, lo
  que convierte a H6 en una de las hipotesis mejor respaldadas. La intuicion es
  que el estilo agresivo de Twitter (mayusculas sostenidas, signos repetidos,
  alargamiento vocalico) es una senal de intensidad afectiva independiente del
  lexico.
- **Experimento que la probaria**: **E06** `experimento_06_estilometria.py`.
  Ablacion de tres sub-bloques: (1) palabras funcionales, (2) puntuacion,
  (3) medidas de forma continuas. Ademas, una variante "contenido enmascarado"
  al estilo de Stamatatos (2017): char n-grams sobre el texto con las palabras
  de contenido sustituidas por una marca generica, para medir el aporte de pura
  forma.
- **Familia de features**: estilometria de superficie y funcional.
- **Criterio de falsacion**: si ninguno de los tres sub-bloques mejora la metrica
  primaria, H6 se rechaza. Si mejora solo el sub-bloque (2), la conclusion es que
  la senal es de intensidad, no de estilo, y se documenta asi.
- **Riesgo**: medio, pero el respaldo en este corpus concreto es fuerte.

---

## H7 - La lematizacion no aporta en tweets

- **Enunciado**: sustituir los tokens por sus lemas (spaCy, D-003) no mejora el
  F1 y puede empeorarlo, comparado con usar las formas de superficie.
- **Evidencia de la literatura**:
  - Stamatatos (2009, DOI 10.1002/asi.21001): en textos cortos y ruidosos, la
    lematizacion y el stemming **a menudo no aportan** frente a los n-gramas de
    caracter y las palabras funcionales.
  - Nina-Alcocer (2019): analizo explicitamente lematizacion y stemming entre sus
    variantes de preprocesamiento, sin que se convirtieran en la configuracion
    ganadora.
  - Wiegand et al. (2018, DOI 10.18653/v1/N18-1095): las features mas utiles para
    abuso fueron POS y sintaxis sobre la forma, no sobre el lema.
- **Razonamiento propio**: en este corpus hay un argumento adicional y decisivo:
  el texto esta **normalizado a ASCII sin diacriticos** (D-010). Un lematizador
  entrenado en espanol con diacriticos puede producir lemas incorrectos sobre
  formas degradadas ("organizacin" -> lema erroneo), introduciendo ruido en lugar
  de reducir la dispersion. La lematizacion tampoco ayuda a los n-gramas de
  caracter, que son la representacion principal de H1.
- **Experimento que la probaria**: **E07** `experimento_07_lematizacion.py`.
  Tres variantes con representacion y clasificador fijos:
  (a) formas de superficie, (b) lemas, (c) formas de superficie + POS como tokens
  adicionales. Se mide tambien la tasa de lemas invalidos o vacios como
  diagnostico del argumento de los diacriticos.
- **Familia de features**: preprocesamiento (tokenizacion / normalizacion).
- **Criterio de falsacion**: si (b) supera a (a) en la metrica primaria, H7 se
  rechaza y la lematizacion pasa al pipeline principal. Es una hipotesis
  **negativa** (predice ausencia de mejora), asi que el resultado informativo es
  tanto confirmarla como refutarla.
- **Riesgo**: bajo. El valor de esta hipotesis no es el F1 sino **simplificar el
  pipeline**: si se confirma, se elimina un paso costoso y una fuente de ruido, y
  el notebook gana en claridad.

---

## H8 - El post-proceso jerarquico (HS=0 => TR=0, AG=0) mejora el F1

- **Enunciado**: forzar la consistencia estructural de las predicciones
  (si HS=0 entonces TR=0 y AG=0) mejora la metrica primaria respecto de umbralizar
  cada clasificador de forma independiente.
- **Evidencia de la literatura**:
  - Silla & Freitas (2011, `silla2011survey`, DOI 10.1007/s10618-010-0175-9):
    la clasificacion jerarquica con **restricciones de consistencia** (una clase
    hija no puede activarse si la padre esta inactiva) suele mejorar el
    rendimiento frente a tratar las etiquetas como independientes. Sustento
    teorico directo.
  - Basile et al. (2019, DOI 10.18653/v1/S19-2007): en HatEval, TR y AG se
    anotaron **solo sobre tweets de odio**. La estructura condicionada es una
    propiedad del esquema de anotacion, no una suposicion nuestra.
  - Fortuna et al. (2019, `fortuna2019hierarchically`, DOI 10.18653/v1/W19-3510):
    corpus con etiquetas organizadas jerarquicamente; precedente de diseno.
  - **Evidencia empirica propia** (Fase 0): TR=1 solo ocurre con HS=1 en 519/519
    casos (0 violaciones); AG=1 con HS=0 en 26 casos (1.5 %). La estructura esta
    en los datos, no solo en la teoria. Decidido como D-001.
- **Razonamiento propio**: el post-proceso elimina falsos positivos de TR y AG
  en los casos donde el modelo de HS predice 0. Su efecto sobre F1 depende de dos
  cantidades medibles: cuantos TR/AG positivos verdaderos tienen HS=0 (casi
  ninguno: 0 y 26) y cuantos TR/AG falsos positivos tienen HS=0 (potencialmente
  muchos). Si la segunda supera a la primera, el F1 de TR y AG **sube**; el de
  HS no cambia. Es por tanto una hipotesis casi determinista y verificable con la
  matriz de confusion de validacion.
- **Experimento que la probaria**: **E08** `experimento_08_postproceso_jerarquico.py`.
  Tres variantes sobre las mismas probabilidades out-of-fold: (a) umbral 0.5
  independiente por etiqueta, (b) post-proceso jerarquico duro, (c) post-proceso
  suave (multiplicar la probabilidad de TR y AG por la probabilidad de HS, en
  lugar de anularla). Se reporta la matriz de confusion 3x3x3 agregada y el
  cambio en cada etiqueta por separado.
- **Familia de features**: post-proceso sobre las predicciones (no usa features).
- **Criterio de falsacion**: si la variante (b) no mejora la metrica primaria, H8
  se rechaza. Dado el patron de los datos esto solo puede ocurrir si el modelo de
  HS produce muchisimos falsos negativos, en cuyo caso el hallazgo importante es
  ese, y el analisis se desplaza al recall de HS.
- **Riesgo**: muy bajo. Coste computacional casi nulo, respaldo empirico directo.

---

## H9 - Optimizar un umbral por etiqueta (distinto de 0.5) mejora el F1

- **Enunciado**: elegir el umbral de decision de cada etiqueta maximizando F1
  sobre validacion (no sobre test) mejora la metrica primaria respecto de usar
  0.5 en las tres.
- **Evidencia de la literatura**:
  - Lipton et al. (2014, `lipton2014thresholding`, DOI
    10.1007/978-3-662-44851-9_15, arXiv:1402.1892): el umbral que maximiza F1 se
    calcula de forma **exacta y eficiente** a partir de la curva de rendimiento
    del clasificador; 0.5 casi nunca es optimo bajo desbalanceo.
  - Nan et al. (2012, `nan2012optimizing`, arXiv:1206.4625): compara el enfoque
    de umbral con el plug-in; el de umbral es robusto y barato.
  - MacAvaney et al. (2019, DOI 10.1371/journal.pone.0221152): identifica el
    desbalanceo de clases como uno de los retos centrales; el umbral es la
    respuesta mas directa.
  - Buda et al. (2018, DOI 10.1016/j.neunet.2018.07.011): el efecto del
    desbalanceo depende de su grado; con desbalanceo moderado las soluciones
    suaves (ponderacion y umbral) son preferibles al remuestreo agresivo.
- **Razonamiento propio**: nuestras tres etiquetas tienen prevalencias distintas
  (43.67 %, 29.98 %, 38.24 %). Un umbral unico de 0.5 es optimo solo si las
  clases estuvieran balanceadas y las probabilidades bien calibradas. Ademas,
  como TR tiene solo 519 positivos, la curva de F1 frente al umbral sera mas
  plana y el umbral optimo mas inestable: por eso hay que elegirlo con
  **validacion cruzada anidada** (elegir el umbral en los folds internos y
  medirlo en los externos) y no con un unico split, para no sobreajustar el
  umbral al conjunto de validacion.
- **Experimento que la probaria**: **E09** `experimento_09_umbral_optimo.py`.
  Para cada etiqueta: curva de F1 frente al umbral en pasos de 0.01 sobre las
  probabilidades out-of-fold, seleccion por maximo de F1 y por indice de Youden,
  y evaluacion honesta con validacion anidada. Se comparan las tres estrategias
  (0.5 fijo, umbral por etiqueta, umbral global compartido). Se reporta tambien
  la calibracion (Brier score y curva de fiabilidad) porque un umbral optimizado
  sobre probabilidades mal calibradas no transfiere bien al test.
- **Familia de features**: post-proceso sobre las predicciones.
- **Criterio de falsacion**: si el umbral optimizado no mejora la metrica primaria
  en validacion anidada, H9 se rechaza. El riesgo real no es que no mejore sino
  **sobreajustar el umbral**: por eso la validacion anidada es obligatoria y el
  umbral se recalcula con todos los datos de train al generar la submission.
- **Riesgo**: muy bajo en coste, bajo en riesgo si se hace anidado.

---

## H10 - El stacking con predicciones out-of-fold supera al mejor modelo individual

- **Enunciado**: un meta-clasificador entrenado sobre las probabilidades
  out-of-fold de varios modelos base heterogeneos (SVM lineal, regresion
  logistica, complemento de Naive Bayes, LightGBM sobre features densas) mejora
  la metrica primaria respecto del mejor modelo base individual.
- **Evidencia de la literatura**:
  - Wolpert (1992, `wolpert1992stacked`, DOI 10.1016/S0893-6080(05)80023-1): un
    meta-aprendiz de segundo nivel sobre predicciones out-of-fold supera a la
    combinacion por votacion simple.
  - **Contraevidencia obligatoria y muy informativa**: Plaza-del-Arco et al.
    (2019, `plazadelarco2019sinai`, DOI 10.18653/v1/S19-2084) aplicaron un
    ensemble por **votacion** de SVM, regresion logistica y arboles sobre este
    mismo corpus y obtuvieron 0.707 frente a 0.701 del baseline SVC en espanol
    (+0.006, practicamente nulo). Es evidencia directa de que la votacion no
    funciona aqui.
  - Argota Vega et al. (2019): su ganancia no vino de ensamblar modelos sino de
    **buscar en el espacio de features** con un unico clasificador. Sugiere que
    el esfuerzo rinde mas en la representacion que en el ensamble.
  - Joachims (1998, DOI 10.1007/BFb0026683): un SVM lineal con muchas features es
    ya muy fuerte en texto, lo que deja poco margen a la combinacion.
- **Razonamiento propio**: la evidencia convergente (SINAI en este corpus +
  MineriaUNAM ganando con un solo SVM + la teoria de Joachims) sugiere que H10 es
  la hipotesis con **menor probabilidad de exito**. Se incluye de todos modos
  porque (i) el stacking es metodologicamente distinto de la votacion y merece
  una prueba honesta, y (ii) un resultado negativo bien documentado es un
  entregable valioso para el notebook. Se probara **despues** de agotar las
  familias de features, no antes.
- **Experimento que la probaria**: **E10** `experimento_10_stacking.py`.
  Generacion de predicciones out-of-fold con la misma particion agrupada para
  todos los modelos base (condicion imprescindible: sin esto hay fuga y el
  resultado es invalido). Meta-modelos comparados: regresion logistica sobre
  probabilidades, SVM sobre probabilidades, y media simple como control. Se
  reporta el F1 de cada base y del meta, por etiqueta.
- **Familia de features**: combinacion de modelos (no de features).
- **Criterio de falsacion**: si el mejor meta-modelo no supera al mejor base en
  la metrica primaria, H10 se rechaza y se documenta como resultado negativo
  convergente con SINAI. Criterio de parada: no se dedica mas de un experimento a
  esta linea.
- **Riesgo**: medio en coste, alto en probabilidad de resultado nulo.

---

## Trazabilidad hipotesis -> experimento -> spec

| Hip. | Experimento | Script | Spec que lo gobierna |
|---|---|---|---|
| H1 | E01 | `experimento_01_char_vs_word_ngrams.py` | `plan/05_modelamiento.md` |
| H2 | E02a / E02b | `experimento_02_lexico_agresividad.py` | `plan/04_procesamiento.md` |
| H3 | E03 | `experimento_03_targeting_tr.py` | `plan/04_procesamiento.md` |
| H4 | E04 | `experimento_04_union_word_char.py` | `plan/05_modelamiento.md` |
| H5 | E05 | `experimento_05_sentimiento_emocion.py` | `plan/04_procesamiento.md` |
| H6 | E06 | `experimento_06_estilometria.py` | `plan/04_procesamiento.md` |
| H7 | E07 | `experimento_07_lematizacion.py` | `plan/04_procesamiento.md` |
| H8 | E08 | `experimento_08_postproceso_jerarquico.py` | `plan/07_mejora.md` |
| H9 | E09 | `experimento_09_umbral_optimo.py` | `plan/07_mejora.md` |
| H10 | E10 | `experimento_10_stacking.py` | `plan/07_mejora.md` |

## Reglas transversales a todos los experimentos

- **Always**: misma particion `StratifiedGroupKFold(k=5)` con grupos de
  near-duplicados en todos los experimentos comparables; mismo `random_state`;
  reportar las cuatro metricas (column-wise, macro-6, micro, exact-match);
  reportar el F1 **por etiqueta** y no solo el agregado; registrar el resultado
  en `experiments.csv` y en `resultados/metricas.json` aunque sea negativo.
- **Ask first**: cualquier uso de un lexico externo nuevo (debe registrarse antes
  en `reports/lexicon_catalog.csv`); cualquier cambio en la particion de
  validacion; cualquier decision que afecte la submission.
- **Never**: usar `toxic_test.csv` para ajustar umbrales, seleccionar features,
  calibrar o elegir hiperparametros; usar etiquetas o corpus externos del esquema
  HatEval (D-002); usar `doc.vector`, `token.vector`, `doc.tensor` o
  `similarity()` de spaCy (D-003); usar transformers o embeddings preentrenados.

## Preguntas abiertas

1. **Que significa TR exactamente en este corpus?** La definicion oficial
   ("dirigido contra una persona o grupo") difiere del esquema HatEval (rango del
   target: grupo frente a individuo). Debe resolverse con el analisis cualitativo
   de 100 ejemplos de E03 antes de optimizar TR (D-005).
2. **Cual es la implementacion real de la metrica?** Pendiente del score publico
   de la sonda E00 que ejecuta Carlos. Las cuatro implementaciones plausibles dan
   valores muy distintos (0.541 / 0.543 / 0.270 / 0.284 con la plantilla todo-1).
3. **Cuanto pesa el pseudo-autor?** 17 pseudo-autores con 3 o mas tweets (4.4 %
   del corpus) y 9 grupos homogeneos en HS. Si la validacion agrupada por
   near-duplicados difiere mucho de la agrupada por pseudo-autor, hay que
   reportar ambas.
4. **Es TR separable en absoluto con features de superficie?** Si E03 falla, la
   respuesta es que no, y la estrategia pasa a depender de la correlacion
   estructural con HS.
