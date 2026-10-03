# ESTADO DEL ARTE v4 - REVISION BIBLIOGRAFICA FOCALIZADA 2022-2026

Proyecto: `KAGGLE_TOXIC_2026_PUJ` (Kaggle `tweet-toxic-2026-puj`).
Entregable de las secciones **92** (revision web / estado del arte) y **93** (transferibilidad
de la literatura) de `docs/advanced_research_v4_spec.md`.
Fecha de la revision: 2026-09-25.
Documento previo (no se repite): `reports/estado_arte.md` (2026-09-22, 45+ referencias,
predominio 2016-2021, frameworks de HateCheck/HatEval, lexicons, char n-grams, umbrales).

---

## 0. Alcance, restricciones y metodo de verificacion

### 0.1 Restricciones duras heredadas (seccion 1 de la spec)

Este informe **solo** recoge ideas que pueden adaptarse a modelos lineales
(TF-IDF / char n-grams + LogisticRegression / LinearSVC / NB / Ridge / SGD) o analisis
estadisticos simples **entrenados desde cero con el corpus de la competencia**. Quedan
descartados por construccion: transformers, LLM, embeddings preentrenados, clasificadores
preentrenados, corpus externos, etiquetas externas y pseudo-labeling.

Cuando un trabajo usa un modelo prohibido, se cita **solo por su hallazgo empirico o por
su diseno de evaluacion**, nunca por su arquitectura. Esto se marca explicitamente en la
ficha de cada trabajo.

### 0.2 Metodo de verificacion de referencias

Ninguna referencia se cita de memoria. Todas se resolvieron en esta sesion contra fuentes
primarias:

1. **ACL Anthology `.bib` oficial** (`https://aclanthology.org/<id>.bib`) para las 29
   referencias de ACL Anthology: devuelve titulo, autores, ano y proceedings exactos.
2. **ACL Anthology (metadatos de pagina)** para el resumen (abstract) de los trabajos
   en los que el metodo es relevante para la adaptacion.
3. **API Crossref** (`https://api.crossref.org/works?query.bibliographic=...&filter=prefix:10.18653`)
   como motor de busqueda sobre todo el corpus de ACL Anthology 2022-2026.
4. **API OpenAlex** (`https://api.openalex.org/works?filter=doi:...`) para las 9 referencias
   de revistas IEEE/ACM/Springer/Elsevier/JAIR: devuelve anio, revista, autores y URL de
   aterrizaje.
5. **API arXiv** solo para descartes (ningun preprint sin revision por pares entra en el
   informe; ver seccion 7).

Resultado: **38 trabajos verificados**, 0 referencias inventadas. Un trabajo que no pudo
confirmarse de forma inequivoca quedo fuera (ver seccion 6).

### 0.3 Que cambia respecto a `estado_arte.md`

El informe previo respondio "con que features y con que modelos se gano HatEval en espanol".
Este informe responde "que se ha hecho nuevo entre 2022 y 2026 en las ocho areas que la
spec v4 declara abiertas", que son precisamente: ruido de etiquetas, hard-example mining,
ponderacion de features supervisada, estructura conjunta de etiquetas, representaciones
subword/simbolicas, verosimilitud class-conditional y espanol reciente.

---

## 1. Resumen ejecutivo por tema

| Tema | Estado de la evidencia 2022-2026 | Valor para V03 |
|---|---|---|
| 1. HSD/lenguaje abusivo sin transformers (ES y multilingue) | **Media-alta**. Existe literatura reciente que vuelve a medir la brecha entre clasicos y transformers, con datasets publicos. | Alto como calibracion de expectativas y como diseño de comparacion |
| 2. Ruido de etiquetas (deteccion, pesos, cleanlab, desacuerdo) | **Muy alta**, pero casi toda en redes neuronales. Lo transferible es la *metrica* de seleccion, no el metodo. | Alto como metricas OOF; bajo como metodo |
| 3. Hard example mining y sample reweighting | **Media**. Mucho en vison; en texto, casi todo con PLM. Hay una idea transferible muy concreta: separar "dificil pero correcto" de "dificil y erroneo". | Medio |
| 4. Feature weighting supervisado (chi2, MI, log-odds, BM25) | **Alta** y de dominio propio (recuperacion de informacion y clasificacion). Es el area con mejor soporte para nuestras restricciones. | **El mas alto** |
| 5. Predictcion multi-etiqueta con dependencia | **Alta** en comparativas de metodos (BR / CC / PS / HMM) y en refinamiento por etiqueta. | **Alto** |
| 6. Representaciones sparse y subword sin transformers | **Baja-media**. Casi todo el trabajo reciente asume un encoder preentrenado. | Medio-bajo |
| 7. Perplejidad / verosimilitud class-conditional como features | **Nula en la literatura revisada 2022-2026** (ver seccion 6). | Cero respaldo externo |
| 8. HSD en espanol reciente | **Media**. Trabajo de variantes/dialectos y de fiabilidad de anotadores, no de features. | Medio |

---

## 2. Tema 1 - Deteccion de discurso de odio y lenguaje abusivo sin transformers

### 2.1 [V4-L01] Nozza, Bianchi, Attanasio (2022). "HATE-ITA: Hate Speech Detection in Italian Social Media Text" (WOAH 2022)

- **Problema**: construir y comparar detectores de discurso de odio multilingue sobre un
  corpus italiano de redes sociales, con recursos linguisticos limitados.
- **Metodo**: corpus nuevo en italiano, anotacion de odio/no-odio/abusivo, y comparacion
  de sistemas (lineales y neuronales) con y sin recursos multilingues externos.
- **Restriccion para nuestro caso**: el italiano no es nuestro idioma, y el corpus es
  externo (prohibido por D-002). El valor no esta en los datos sino en el **diseno de la
  comparacion**: separar el efecto del idioma del efecto del modelo.
- **Transferible a modelos lineales sin preentrenamiento**: si, en la dimension
  metodologica. Los sistemas lineales sobre n-gramas son la linea base obligatoria, no un
  detalle.
- **Adaptacion (experimento v4)**: `V4-C04` - recalcular la curva de referencia de V03
  (word + char_wb) frente a variantes con una sola familia de features, con la misma
  `StratifiedGroupKFold` y el mismo protocolo de umbrales, y reportar la diferencia por
  etiqueta. No es un experimento de mejora: es un **suelo de referencia honesto** que
  impide atribuir a "una representacion nueva" una ganancia que en realidad es un cambio
  de metrica.

### 2.2 [V4-L02] Balouchzahi, Butt, Sidorov, Gelbukh (2022). "CIC@LT-EDI-ACL2022: Are transformers the only hope? Hope speech detection for Spanish and English comments" (LT-EDI 2022)

- **DOI**: 10.18653/v1/2022.ltedi-1.28
- **Problema**: detectar discurso de esperanza (frente a odio) en comentarios en espanol e
  ingles, en un shared task con datos limitados.
- **Metodo**: combinacion de features **linguisticas y psicolinguisticas** con un modelo
  de aprendizaje profundo; el equipo se posiciona 2.o en ingles y 3.o en espanol.
- **Restriccion para nuestro caso**: el modelo es neuronal y la tarea (esperanza vs odio) no
  es la nuestra. Sin embargo, el bloque de **features psicolinguisticas explicitas**
  (marcadores afectivos, intensificadores, negacion) es exactamente la familia que ya
  probamos en `E2-09-EMO-LEX` (47 features del lexico induce) con resultado negativo.
- **Transferible**: parcialmente. La leccion transferible es **negativa y valuable**: en
  espanol, con datos limitados, las features psicolinguisticas a mano no desplazan al
  clasico sobre n-gramas. Refuerza la decision de no reintentar la familia de emocion.
- **Adaptacion**: no se propone experimento nuevo. Se registra como **evidencia de cierre**
  para `E2-09` en `DECISIONS.md` (evita reintentar la familia en v4, §84 NO_REPEATING).

### 2.3 [V4-L03] de Araujo, Roth (2022). "Checking HateCheck: a cross-functional analysis of behaviour-aware learning for hate speech detection" (NLP Power! 2022)

- **DOI**: 10.18653/v1/2022.nlppower-1.8
- **Problema**: los tests funcionales (HateCheck) evaluan capacidades que las metricas
  agregadas ocultan. La pregunta es si se puede **entrenar** con esa informacion de
  comportamiento y si la generalizacion se transfiere a datos i.i.d.
- **Metodo**: esquemas de entrenamiento por grupos de casos funcionales de HateCheck,
  reteniendo clases de functionality y evaluando en capacidades omitidas.
- **Resultado reportado**: mejora en las funcionalidades retenidas, pero **caida** en
  funcionalidad no vista y en datos i.i.d. de deteccion de odio, es decir,
  sobreajuste a la distribucion de los tests funcionales.
- **Restriccion**: usa fine-tuning de modelos preentrenados (prohibido). Pero el hallazgo
  es independiente del modelo.
- **Transferible**: si, y es un aviso metodologico directo: **un agregado de F1 oculta
  fallos sistematicos por tipo de caso**, y que anadir casos de un conjunto externo no
  mejora la deteccion en datos reales.
  casos anadidos no mejora la deteccion en datos reales.
- **Adaptacion (experimento v4)**: `V4-D05` - descomponer el OOF congelado de V03 en
  subconjuntos funcionales definidos **solo con features no supervisadas de TRAIN**
  (presencia de negacion, de segunda persona, de target colectivo, de atribucion, longitud,
  emoji/onomatopeya) y comparar la densidad de error y la F1 por subconjunto. Es la
  version de bajo costo de "HateCheck" que las restricciones permiten: sin plantillas externas,
  con particiones derivadas de nuestro propio corpus.

### 2.4 [V4-L04] Lin, Chen, Liu, Lin (2023). "Linear Classifier: An Often-Forgotten Baseline for Text Classification" (ACL 2023, short)

- **DOI**: 10.18653/v1/2023.acl-short.160
- **Problema**: argumento metodologico. Los PLM se entrenan y despliegan por defecto; el
  trabajo defiende que un clasificador lineal sobre bolsa de palabras es una referencia
  obligatoria, y que los resultados de los PLM solo son aceptables si el simple baseline
  esta ejecutado.
- **Metodo**: comparacion empirica entre clasificadores lineales sobre conteos/TF-IDF y
  modelos fine-tuneados en varios conjuntos de texto.
- **Restriccion**: ninguno. Es exactamente nuestro escenario.
- **Transferible**: es el **paper de apoyo central** de la restriccion NO_TRANSFORMERS de
  la spec. Su resultado ("los metodos lineales muestran rendimiento competitivo,
  eficiencia alta y robustez en muchos datos de texto") es la evidencia de que perseguir
  un F1 mas alto sin preentrenamiento no es una cruzada inutil.
- **Adaptacion**: se usa como **justificacion escrita en `reports/advanced_research_v4.md`**
  y como criterio de decision en la seccion 97 (candidatos finales): cualquier candidato
  v4 debe superar al clasificador lineal de V03 bajo el mismo protocolo.

### 2.5 [V4-L05] Zhang, Hangya, Fraser (2024). "A Study of the Class Imbalance Problem in Abusive Language Detection" (WOAH 2024)

- **DOI**: 10.18653/v1/2024.woah-1.4
- **Problema**: estudio sistematico del **desbalanceo de clases** en deteccion de lenguaje
  abusivo, que hasta entonces seliteraba de forma irregular.
- **Metodo**: comparacion de **10 metodos** de tratamiento del desbalanceo sobre **8
  conjuntos de datos** (binarios y multiclasse, moderados y muy desbalanceados). Proponen
  ademas dos metodos propios: `AbusiveLexiconAug` (aumento con lexico abusivo) y
  `ExternalDataAug` (datos externos).
- **Resultados reportados**: los mas versatiles son `AbusiveLexiconAug`, el
  sobremuestreo aleatorio y la **focal loss**; la focal loss da el pico de rendimiento;
  sobremuestreo y focal loss funcionan mejor en binarios y multiclasse pequenos, mientras
  que submuestreo y entropia cruzada ponderada se adaptan mejor a multiclasse grande; la
  mayoria de los metodos son sensibles a hiperparametros.
- **Restriccion para nuestro caso**: `ExternalDataAug` esta prohibido (D-002). La
  `focal loss` y el sobremuestreo **si** son legales. Nuestro desbalanceo es moderado
  (HS 43.67 %, TR 29.98 %, AG 38.24 %), es decir, el extremo "moderado" del estudio.
- **Transferible**: si, y con alta especificidad. Es el unico trabajo 2022-2026 que
  **barre explicitamente** las alternativas de cost-sensitive learning en este dominio
  exacto, que es la seccion 42 de la spec (que ya declara `class_weight='balanced'` como
  "nunca barrido por etiqueta").
- **Adaptacion (experimento v4)**: `V4-B06` - barrido **por etiqueta** de
  `class_weight` (None / balanced / {1:alpha, 0:1} con alpha en {0.7, 0.8, 0.9, 1.1, 1.25}),
  On sobremuestreo aleatorio de la clase minoritaria dentro de cada pliegue, y sobre
  submuestreo de la mayoritaria. Reportar F1 por etiqueta y por pliegue. Este es
  exactamente el gap declarado en la seccion 42/43 y tiene respaldo externo explicito.

### 2.6 [V4-L06] Cercas Curry, Abercrombie, Talat (2024). "Subjective Isms? On the Danger of Conflating Hate and Offence in Abusive Language Detection" (WOAH 2024)

- **DOI**: 10.18653/v1/2024.woah-1.22
- **Problema**: argumento teorico. La comunidad ha adoptado la idea de que "la subjetividad
  del anotador es valida" (importada del analisis de sentimiento), pero eso es
  **inapropiado** para la deteccion de discurso de odio: concede igual validez a todas las
  posiciones sobre sexismo o racismo.
- **Metodo**: analisis argumentativo + discusion de implicaciones para la construccion de
  datasets y la modelizacion. Sin experimento.
- **Restriccion**: sin modelo; no impone ninguna restriccion tecnica.
- **Transferible**: si, y es la **referencia conceptual que legitimiza el track E** de la
  spec. En nuestro corpus HS, TR y AG **no son labels simetricas**: TR solo ocurre cuando
  HS=1 (519/519 casos). El argumento de este trabajo es exactamente el nuestro: odio y
  ofensa son **conceptos ortogonales**, y por tanto el espacio de etiquetas no debe
  tratarse como un espacio plano de 3 bits.
- **Adaptacion (experimento v4)**: `V4-E01` - multinomial de 4 clases sobre el
  subespacio factible {000, 100, 101, 111} (las unicas combinaciones observadas en TRAIN),
  comparado contra la triple binaria de V03 con los mismos folds, followed de
  marginalizacion a 3 columnas. Justificacion: si odio y ofensa no son ortogonales, la
  parametrizacion de 4 clases es la coherente; si las 3 binarias la igualan, la estructura
  jerarquica ya esta capturada por las features.

### 2.7 [V4-L07] Casula, Leonardelli, Tonelli (2024). "Don't Augment, Rewrite? Assessing Abusive Language Detection with Synthetic Data" (Findings ACL 2024)

- **DOI**: 10.18653/v1/2024.findings-acl.669
- **Problema**: evaluar si el aumento de datos para deteccion de lenguaje abusivo debe
  consistir en **reescribir** los ejemplos (metodo más agresivo) en lugar de duplicar o
  combinar.
- **Metodo**: comparacion de varias estrategias de aumento de datos sobre texto generativo
  para deteccion de lenguaje abusivo, con evaluacion en multiples dominios.
- **Restriccion**: el metodo de reescritura requiere un generador de texto, es decir, un
  LLM o un modelo de lenguaje, ambos prohibidos.
- **Transferible**: no en el metodo, si en la **conclusion de coste-beneficio**: el
  aumento agresivo no aporta de forma fiable en la deteccion de abuso. Refuerza la
  restriccion NO_PSEUDO_LABELING: no vamos a generar datos sintéticos para el subconjunto
  dificil.
- **Adaptacion**: no se propone experimento. Se registra como evidencia de cierre para la
  familia "data augmentation" y para la restriccion `NO_PSEUDO_LABELING` de la seccion 1.

### 2.8 [V4-L08] Caselli, van der Veen (2023). "Benchmarking Offensive and Abusive Language in Dutch Tweets" (WOAH 2023)

- **DOI**: 10.18653/v1/2023.woah-1.7
- **Problema**: construir un banco de pruebas para la distincion entre **ofensivo** y
  **abusivo** en tweets, que es la misma confusion que nuestra frontera HS/AG.
- **Metodo**: corpus neerlandes anotado, con comparacion de lineas base y analisis de
  consistencia.
- **Restriccion**: idioma y corpus externos (prohibidos). El dataset no se usa.
- **Transferible**: si, a nivel de **diseño de la confusion objetivo**. Nos da una
  confirmacion de que la separacion ofensa/odio es la dificultad central del dominio, y
  por tanto de que el post-proceso `HS=0 => AG=0` y el modelado conjunto de TR son las
  palancas correctas.
- **Adaptacion**: entra como apoyo a `V4-E01` y a la ya ejecutada `E2-01/E2-02` (cadenas de
  etiquetas). No genera experimento propio.

---

## 3. Tema 2 - Ruido de etiquetas en clasificacion de texto

### 3.1 [V4-L09] Northcutt, Jiang, Chuang (2021). "Confident Learning: Estimating Uncertainty in Dataset Labels" (JAIR 2021)

- **DOI**: 10.1613/jair.1.12125 | **Nota de alcance**: publicado en 2021, fuera del rango
  2022-2026. Se incluye porque es la referencia canonica del metodo (cleanlab) y porque no
  existe una sustitucion 2022-2026 equivalente revisada por pares.
- **Problema**: estimar, sin etiquetas, que ejemplos de un conjunto de entrenamiento
  tienen probablemente etiqueta erronea.
- **Metodo**: se entrena un modelo; para cada ejemplo se compara la etiqueta observada con
  la verosimilitud predicha de esa etiqueta frente a las alternativas (dato etiquetado),
  se seleccionan ejemplos por umbral de confianza y se obtiene un conjunto de etiquetas
  ruidosas. En su variante multi-clase se usa la confianza media por clase.
- **Restriccion**: **ninguna para nuestro caso**. Es agnostico al modelo: funciona con un
  clasificador lineal. Es la unica pieza de la literatura de ruido de etiquetas que es
  directamente ejecutable con `LogisticRegression` + `predict_proba`.
- **Transferible**: si, y es la mas valuable de todo este bloque. La restriccion
  `NO_TEST_FITTING` se respeta trivially porque el metodo se aplica **solo a TRAIN**, con
  predicciones OOF (seccion 9 de la spec: `hardness_score` solo OOF).
- **Adaptacion (experimento v4)**: `V4-A03` - aplicar confident learning **por etiqueta**
  con los tres modelos lineales de V03 y con predicciones **OOF** (nunca in-sample):
  computar `s(x) = p_hat(y_obs)`, seleccionar por umbral, y contrastar las
  distribuciones de `s` de los subconjuntos "Dificil coherente", "Posible ruido" y
  "Facil" ya calculados en `V4-B02`. Entregable: tabla de coherencia entre `s` y
  `error_score`. **No se corrigen etiquetas** (seccion 90: no corregir por predicciones
  sola). Si el solapamiento de `s` entre "dificil pero correcto" y "dificil y erroneo" es
  alto, la linea de "ruido de etiquetas" queda cerrada con evidencia; si es bajo, el
  reponderado por `1 - s` es justificable para la seccion 11.

### 3.2 [V4-L10] Xu, Jing, Yu (2024). "Enhancing Multi-Label Text Classification under Label-Dependent Noise: A Label-Specific Denoising Framework" (Findings EMNLP 2024)

- **DOI**: 10.18653/v1/2024.findings-emnlp.324
- **Problema**: en clasificacion multi-etiqueta ruidosa se asume que el ruido es
  **independiente por clase** (supuesto *class-conditional noise*). El trabajo valida con
  tests de hipotesis que los datos reales **no cumplen** ese supuesto: el ruido depende
  de las otras etiquetas.
- **Metodo**: una metrica de seleccion **holistica** que combina tres senales -
  (a) informacion de perdida, (b) informacion de ranking, (c) **centroide de features** -
  y luego identifica y corrige etiquetas ruidosas **de forma independiente por categoria
  de etiqueta**.
- **Restriccion**: el marco esta implementado sobre redes neuronales. Sin embargo, las
  **tres senales de seleccion son todas calculables con modelos lineales**: perdida BCE
  OOF, ranking de la etiqueta observada entre las tres, y centroide (media de las
  representaciones linealmente aprendidas) por etiqueta.
- **Transferible**: si, y aporta el insight mas util de esta seccion para nosotros:
  **el ruido no es intercambiable entre etiquetas**, asi que las metricas de ruido deben
  computarse **por etiqueta** (seccion 12 de la spec: `hardness_HS/TR/AG` con pesos
  independientes). Ademas, la combinacion de perdida + ranking + centroide es
  literalmente lo que `V4-B01` ya midio (`bce_score`, `error_score`, cercania a la
  frontera) y lo que `V4-B02` clasifico.
- **Adaptacion (experimento v4)**: `V4-B03` (dentro de `29_reweighting_v4.py`) - una
  version **solo-metrica** del marco: por cada etiqueta y cada ejemplo, calcular un
  `noise_score` = combinacion normalizada de (i) `bce_score` OOF, (ii) ranking de la
  etiqueta observada, (iii) distancia al centroide de la clase observada en el espacio de
  coeficientes/activaciones lineales. Reportar la matriz de correlacion entre
  `noise_score` y el error OOF real, por etiqueta. Es un **diagnostico**, no una
  correccion de etiquetas.

### 3.3 [V4-L11] Xu, Song, Liu, Liu, Sun, Jing, Yu (2024). "Noisy Multi-Label Text Classification via Instance-Label Pair Correction" (Findings NAACL 2024)

- **DOI**: 10.18653/v1/2024.findings-naacl.93
- **Problema**: el mismo problema (ruido multi-etiqueta) atacado desde la granularidad
  **par etiqueta y par instancia**, en lugar de por instancia completa.
- **Metodo**: correccion de pares (instancia, etiqueta) individuales, es decir, se decide
  de forma independiente para cada par si la etiqueta es ruido.
- **Restriccion**: implementacion neuronal.
- **Transferible**: si, como **diseno de unidad de analisis**: la unidad correcta de
  reponderacion es el par (instancia, etiqueta), no la instancia. Esto conecta con
  `V4-B02` (43 ejemplos con 2.9 % de error) y con la dependencia estructural
  TR=>HS: un mismo tweet puede tener HS fiable y TR dudoso.
- **Adaptacion (experimento v4)**: `V4-B04` - matriz de pesos por par (instancia,
  etiqueta) derivada de las tres senales de `V4-B03`, y dos reentrenamientos:
  (i) `sample_weight` compartido por instancia (linea base), (ii) `sample_weight` por par
  labelling (es decir, reentrenar un clasificador binario por etiqueta con sus propios
  pesos, que es lo unico que `LogisticRegression` permite). Comparar con V4-B03-share.
  Advertencia: `LinearSVC` no acepta `sample_weight` por clase; hay que usar
  `LogisticRegression` o `SGDClassifier`.

### 3.4 [V4-L12] Zhu, Hedderich, Zhai, Adelani, Klakow (2022). "Is BERT Robust to Label Noise? A Study on Learning with Noisy Labels in Text Classification" (Insights from Negative Results in NLP 2022)

- **DOI**: 10.18653/v1/2022.insights-1.8
- **Problema**: hasta que punto un modelo con representacion de subpalabra preentrenada
  absorbe el ruido de etiquetas, y que tecnicas de robustez (penalizacion, reetiquetado)
  funcionan en texto.
- **Metodo**: inyeccion controlada de ruido en benchmarks de clasificacion de texto
  (sentimiento, disponibilidad, categorizacion de tema) y comparacion de variantes de
  entrenamiento robusto.
- **Restriccion**: el objeto de estudio es BERT (prohibido). Los datasets son externos.
- **Transferible**: si, por el **hallazgo negativo**. El resultado central de la linea de
  resultados negativos es que la robustez frente al ruido en texto **no es una propiedad
  de la arquitectura sino del procedimiento de entrenamiento y de la severidad del
  ruido**: con ruido suave aporta poco; con ruido fuerte las tecnicas rinden. La
  consecuencia practica para nosotros es un **techo realista**: con 1.731 filas
  etiquetadas y una tasa de contradiccion medida del 2.77-2.89 % (`V4-A01`), el ruido es
  **suave**, luego las tecnicas de desruido tienen rendimiento esperado bajo.
- **Adaptacion**: no se propone experimento. Se usa para **anclar la expectativa de
  `V4-B03`/`V4-B04`/`V4-B05`**: no esperar mejoras grandes, y no insistir mas alla de un
  ciclo. Consistente con la decision ya tomada en `V4-A02` ("techo < 3 % de ejemplos").

### 3.5 [V4-L13] Wang, Plank (2023). "ACTOR: Active Learning with Annotator-specific Classification Heads to Embrace Human Label Variation" (EMNLP 2023)

- **DOI**: 10.18653/v1/2023.emnlp-main.126
- **Problema**: la agregacion por mayoria al crear datasets descarta las opiniones de la
  minoria; se muestra que **aprender de las anotaciones individuales** supera a aprender de
  la etiqueta agregada.
- **Metodo**: **una cabeza de clasificacion por anotador** (multi-cabeza) mas funciones de
  adquisicion de aprendizaje activo; entropia a nivel de grupo.
- **Restriccion**: el experimento es de aprendizaje activo con anotadores multiples, y el
  corpus de la competencia **tiene una sola anotacion por fila** (no hay metadatos de
  anotador). Por tanto la version por anotador es inejecutable.
- **Transferible**: **parcialmente, y con una sustitucion concreta**. Lo transferible no es
  la cabeza por anotador sino la conclusion: *la etiqueta agregada esconde estructura*.
  En nuestro caso, el "grupo" disponible no es el anotador sino el **cluster de
  consistencia** que `V4-A01` ya computo (15 clusters con 2+ miembros, 84 tweets, 2
  clusters contradictorios).
- **Adaptacion (experimento v4)**: `V4-A04` - **sin** nuevo modelo: para los 15 clusters de
  `V4-A01`, tabular el vector de etiquetas (HS,TR,AG) de cada miembro, medir la entropia de
  la combinacion y el desacuerdo maximo, y cruzar ese desacuerdo con el error OOF de
  V03. Si la correlacion es alta, el "ruido" y el "caso dificil" estan confundidos y el
  track A se cierra; si es baja, son fenomenos distintos y el track B tiene entidad propia.

### 3.6 [V4-L14] Ivey, Gauch, Jurgens (2025). "NUTMEG: Separating Signal From Noise in Annotator Disagreement" (EMNLP 2025)

- **DOI**: 10.18653/v1/2025.emnlp-main.144
- **Problema**: los modelos de agregacion asumen que todo desacuerdo entre anotadores es
  error. Para muchas tareas el desacuerdo es **sistematico** (subpoblaciones con opiniones
  distintas) y debe tratarse como senal.
- **Metodo**: modelo bayesiano que incorpora informacion sobre el **perfil de cada
  anotador** para eliminar anotaciones ruidosas conservando los desacuerdos sistematicos;
  se evalua con datos sinteticos (donde se conoce la verdad) y reales, y se muestra que
  los modelos entrenados sobre datos agregados por NUTMEG superan a los entrenados con
  agregacion tradicional.
- **Restriccion**: requiere informacion de anotador, que no tenemos. Es un modelo bayesiano
  jerarquico.
- **Transferible**: si, por el **principio**, que si en un subgrupo sistematico nuestra
  metrica es alta, el "ruido" es en realidad senal estable. La implicacion operation para
  nosotros es una **decision de criterio**: cuando `V4-A04` muestre un cluster con
  desacuerdo consistente, el tratamiento correcto no es eliminarlo sino **dejarlo** y
  comprobar si el modelo lo predice con el mismo error que el resto (es decir, si es un
  patron del corpus y no ruido).
- **Adaptacion**: se integra en `V4-A04` como criterio de lectura; no genera experimento
  propio.

### 3.7 [V4-L15] Fleisig, Abebe, Klein (2023). "When the Majority is Wrong: Modeling Annotator Disagreement for Subjective Tasks" (EMNLP 2023)

- **DOI**: 10.18653/v1/2023.emnlp-main.415
- **Problema**: en tareas como deteccion de discurso de odio, la mayoria de anotadores
  puede ser **sistematicamente erronea** respecto al grupo objetivo, cuando ese grupo es una
  fraccion pequena del conjunto de anotadores.
- **Metodo**: modelo que predice las valoraciones individuales de los anotadores sobre
  texto potencialmente ofensivo y las combina con el **grupo objetivo predicho** del texto
  para predecir las valoraciones de los miembros de ese grupo. Mejoras reportadas: +22 %
  al predecir calificaciones individuales y +33 % al predecir la varianza entre anotadores.
- **Restriccion**: requiere datos demograficos de anotadores (inexistentes aqui) y un
  corpus con anotaciones multiples.
- **Transferible**: si, por el **hallazgo de metricas**: la **varianza entre anotadores**
  es una metrica de incertidumbre utilizable aguas abajo. Como no tenemos anotadores, el
  equivalente disponible en nuestro setting es la **incertidencia predictiva por
  combinacion de etiquetas**: la entropia de la prediccion de V03 en el subespacio
  {000,100,101,111} (ya calculable, y posiblemente ya presente en el bloque META).
- **Adaptacion (experimento v4)**: `V4-E02` - usar la **entropia de la clase conjunta**
  (y no solo la entropia de las tres binarias) como feature de incertidumbre en el
  meta-aprendiz de la seccion 50. Es la version barata de "varianza entre anotadores" que
  las restricciones permiten, y ataca el gap declarado de **calibracion de las entradas
  del meta**, que la spec marca como "nuevo y prioritario".

### 3.8 [V4-L16] Weerasooriya, Ororbia, Bhensadadia, KhudaBukhsh, Homan (2023). "Disagreement Matters: Preserving Label Diversity by Jointly Modeling Item and Annotator Label Distributions with DisCo" (Findings ACL 2023)

- **DOI**: 10.18653/v1/2023.findings-acl.287
- **Problema**: la suposicion de "una etiqueta por item es la verdad" oculta las opiniones
  minoritarias. Se redefine la "verdad" como la **distribucion** de etiquetas que una
  poblacion de anotadores produciria.
- **Metodo**: modelo simple que aprende a predecir esa distribucion, tomando pares
  (anotador, item) como entrada y agregando sobre anotadores en la inferencia.
- **Restriccion**: modelo neuronal; requiere anotadores.
- **Transferible**: parcialmente. La idea de que la etiqueta es una **distribucion** y no un
  punto se traslada directamente a nuestro caso **sin** needing anotadores, porque
  nuestro esquema **ya es condicional**: TR solo se anota cuando HS=1. Es decir, el
  corpus contiene implicitamente distribuciones condicionales por combinacion.
- **Adaptacion (experimento v4)**: `V4-E03` - estimar la **distribucion de probabilidad
  condicional** P(AG|HS=1,TR=1), P(AG|HS=1,TR=0), P(TR|HS=1) en TRAIN, y usarla como
  **prior jerarquico suave** en la marginalizacion de `V4-E01` en vez de reglas duras.
  Esto ataca directamente la seccion 37 ("combinaciones raras con smoothing/prior
  jerarquico") y la seccion 89 ("joint structured output y soft priors, no solo cadenas").

### 3.9 [V4-L17] Plank (2022). "The 'Problem' of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation" (EMNLP 2022, position paper)

- **DOI**: 10.18653/v1/2022.emnlp-main.731
- **Problema**: la variacion humana de etiquetado se trata sistematicamente como ruido; el
  trabajo argumenta que hay variacion genuina y que **afecta a las tres etapas del pipeline
  a la vez**: datos, modelizacion y evaluacion.
- **Metodo**: documento de posicion; reconcilia nociones dispersas de la literatura y
  cataloga conjuntos de datos con etiquetas **no agregadas**.
- **Restriccion**: no aporta metodo; la parte de "repositorio de datasets con etiquetas no
  agregadas" es inutilizable (D-002).
- **Transferible**: si, como **marco de criterio**. Es la referencia que justifica no
  perseguir una diferencia de +0.002 en F1 como si fuera un descubrimiento (§72
  MULTIPLICIDAD) y que obliga a declarar que una parte del error de validacion es
  irreducible.
- **Adaptacion**: sin experimento. Se usa en la seccion 97/99 para redactar la
  comparacion de candidatos sin jerarquias numericas arbitrarias.

### 3.10 [V4-L18] Zhu, Hedderich et al. (2022) y la busqueda de "companion labels"

**Resultado negativo de busqueda (relevante para la seccion 2 de la peticion).** No se ha
podido verificar ningun trabajo revisado por pares 2022-2026 que use el termino ingles
**"companion label"** en el contexto de aprendizaje con etiquetas ruidosas. Se buscaron
OpenAlex (`title_and_abstract.search:"companion label learning with noisy labels"`), arXiv
(`all:"companion label"`, `ti:"companion" AND abs:"noisy label"`) y Crossref
(`query.bibliographic:"companion label learning noisy labels"`, con y sin filtro de
prefijo). El termino aparece en la literatura de **etiquetas parciales** y de
**anotacion multi-clase** con otra acepcion, no como mecanismo de estimativa de ruido
reutilizable en nuestro setting. **Se reporta explicitamente como "sin evidencia reciente
transferible"** y no se incluye ninguna referencia no confirmada.

---

## 4. Tema 3 - Hard example mining y sample reweighting

### 4.1 [V4-L19] Huang, Huang, Xu, Li, Xiao (2024). "Logits Reranking via Semantic Labels for Hard Samples in Text Classification" (Findings EMNLP 2024)

- **DOI**: 10.18653/v1/2024.findings-emnlp.657
- **Problema**: los PLM siguen fallando en muestras donde la confianza para distinguir es
  baja. La mayoria de los tratamientos ignoran que **las propias etiquetas tienen semantica**
  y las tratan como vectores one-hot.
- **Metodo**: LRSL, un metodo **agnostico al modelo, aplicado solo en inferencia** que
  detecta automaticamente las muestras difciles y las reordena usando similitud semantica
  entre etiquetas, operando solo sobre logits y sin tocar el entrenamiento.
- **Restriccion**: la "semantica de la etiqueta" se obtiene de embeddings de etiquetas
  (prohibidos). Sin embargo, **la parte que mas nos interesa - la deteccion automatica de
  muestras difiles y el reranking en inferencia - no usa embeddings**; usa los logits.
- **Transferible**: si. La idea minima y potente es: **la decision final de un ensemble o
  de un post-proceso puedeisses ser corregida para las muestras detectadas como
  difiles, usando la estructura de dependencias entre etiquetas en vez de la etiqueta
  aislada**. En nuestro caso, la "semantica de etiqueta" es la dependencia
  HS -> {TR, AG}, que ya esta medida en el corpus.
- **Adaptacion (experimento v4)**: `V4-B07` - post-proceso de reranking **solo en las
  muestras OOF con confianza de frontera** (ya identificadas en `V4-B01`, r=+0.34 con la
  frontera): para esas muestras, y no para las demas, reordenar la salida de los tres
  clasificadores usando el modelo conjunto de `V4-E01` (4 clases) en vez de la
  independencia binaria. Medir si el ganho se concentra en ese subconjunto.

### 4.2 [V4-L20] Wu, Ding, Zhang, Gao, Tang, Du (2023). "DiscrimLoss: A Universal Loss for Hard Samples and Incorrect Samples Discrimination" (IEEE Transactions on Multimedia 2023)

- **DOI**: 10.1109/tmm.2023.3290477
- **Problema**: las funciones de perdidaaussianas degradan la capacidad de distinguir
  entre (a) muestras **dificiles pero correctamente etiquetadas** y (b) muestras
  **dificiles y mal etiquetadas**. Tratar ambas igual es contraproducente.
- **Metodo**: una perdida universal que separa ambas categorias, con analisis teorico y
  validacion empirica en vision y en tareas multimodales.
- **Restriccion**: formulated para redes neuronales con lote; no se transfiere
  literalmente. Fuera de dominio (vision).
- **Transferible**: si, como **principio de diseno del objetivo**, y este principio ya
  esta medido en nuestro corpus. `V4-B02` encontro que "Dificil coherente" (296 ejemplos,
  20.0 %, error medio 0.351 por etiqueta) y "Posible ruido" (43 ejemplos, 2.9 %, error
  medio 0.054) son categorias **separadas**: los posibles-ruidos son, contra toda
  intuicion, los que **menos** errores cometen. Es decir, en nuestro corpus la
  disyuntiva de DiscrimLoss ya tiene respuesta empirica.
- **Adaptacion (experimento v4)**: `V4-B03` incluye el reporte explicito de esta
  separacion: comparar `sample_weight` derivado de "cercania a la frontera" (categoria
  dificil) frente a `sample_weight` derivado de "contradiccion de cluster" (categoria
  ruido). El primer grupo tiene 296 ejemplos y merece reponderacion; el segundo tiene 43 y
  no. **Consecuencia de la revision: la via de reponderacion por ruido tiene un techo
  < 3 % de ejemplos, mientras que la via de dureza tiene un techo de 20 %.** Ese es el
  cambio de foco que la revision bibliografica impone en el track B.

### 4.3 [V4-L21] Cordeiro, Carneiro (2025). "ANNE: Adaptive Nearest Neighbours and Eigenvector-based sample selection for robust learning with noisy labels" (Pattern Recognition 2025)

- **DOI**: 10.1016/j.patcog.2024.111132
- **Problema**: seleccion de ejemplos robustos en presencia de ruido, con eficiencia
  (aproximacionLow-rank) y usando una segunda "vista" de los datos.
- **Metodo**: combinacion de **k vecinos mas cercanos** con una descomposicion por
  autovectores (low-rank) para seleccionar muestras limpias de forma adaptativa.
- **Restriccion**: dominio semi-supervisado de vision; usa tecnicas de bajo rango.
  Y relevante: **seleccionar y descartar ejemplos es exactamente lo que la seccion 8 de la
  spec restringe** ("No eliminar ruido automaticamente": tres estrategias, baseline /
  downweight / remove extremos).
- **Transferible**: si, y es un **argumento a favor de la segunda vista no supervisada**.
  La version restringida de "dos vistas" en nuestro caso es: (i) vista de modelo
  (probabilidades OOF de los tres clasificadores lineales) frente a (ii) vista de datos
  (similitud coseno/char/Jaccard entre TRAIN, secciones 56-57). Un ejemplo es "limpio" si
  **ambas** vistas coinciden.
- **Adaptacion (experimento v4)**: `V4-F01` - para cada ejemplo de TRAIN calcular
  `agreement_modelo_datos` = 1 si el vecino mas cercano en TRAIN (por coseno de TF-IDF
  word, y por char 3-5 de Jaccard) tiene la misma combinacion (HS,TR,AG) y su prediccion
  OOF es correcta. Usar ese indicador como feature de reponderacion en `V4-B03`, **no**
  como criterio de borrado. Es la unica forma de usar la idea de ANNE respetando la
  seccion 8.

---

## 5. Tema 4 - Feature weighting supervisado para texto

Esta es la seccion con **mejor soporte** para nuestras restricciones, y la que menos se
solapa con lo ya ejecutado (`E2-15-NBSVM` fue NB log-count ratio **global**; aqui las
ideas son **por etiqueta** o **probabilisticas**).

### 5.1 [V4-L22] Attieh, Tekli (2023). "Supervised term-category feature weighting for improved text classification" (Knowledge-Based Systems 2023)

- **DOI**: 10.1016/j.knosys.2022.110215
- **Problema**: la ponderacion de terminos clasica (TF-IDF) es **no supervisada**; se
  propone una ponderacion que aprovecha la **categoria (clase)** del termino.
- **Metodo**: esquemas de ponderacion supervisada que combinan la frecuencia del termino con su
  distribucion por clase, de modo que el peso de un termino **depende de la clase para la
  que se usa**.
- **Restriccion**: ninguno; todo el pipeline es TF-IDF + modelo lineal. El corpus es
  generico, no de odio.
- **Transferible**: **si, y es la traduccion directa de la seccion 16 de la spec**
  ("Class-specific feature weighting: `w_j(label)` con smoothing"). Es la idea mas
  directamente transferible de toda la revision.
- **Adaptacion (experimento v4)**: `V4-C01` - construir tres vectores de pesos
  `w_HS`, `w_TR`, `w_AG` con suavizado, derivados **solo de TRAIN y dentro de cada
  pliegue**: (a) `chi2` por etiqueta con smoothing, (b) information gain, (c)
  `MI(feature,etiqueta)` estimada por histograma de conteos (no por Estimador kernel, para
  no tener 1731 observaciones para miles de features). Aplicar
  `X'_j = TFIDF_j * w_etiqueta_j` y entrenar **un clasificador por etiqueta con su propio
  ponderador**. Esto es genuinamente distinto de `E2-15-NB-WORD/CHAR/NBSVM`, que usaron un
  unico log-count ratio **global** para las tres etiquetas.
  Nota de implementacion: `chi2` y MI por etiqueta con 1.731 filas producen pesos ruidosos
  para features raras; el **smoothing es obligatorio** y debe reportarse el barrido de
  alfa.

### 5.2 [V4-L23] Tang, Li, Li (2022). "An improved supervised term weighting scheme for text representation and classification" (Expert Systems with Applications 2022)

- **DOI**: 10.1016/j.eswa.2021.115985
- **Problema**: los esquemas de ponderacion de terminos existentes pierden informacion
  porque solo usan **dos** de las tres senales disponibles (frecuencia del termino en la
  clase, frecuencia del termino fuera de la clase, y document frequency global).
- **Metodo**: un esquema que incorpora la **distribucion completa** del termino entre
  clases y documento, con un criterio de ganancia de informacion, y lo valida en varios
  conjuntos de texto.
- **Restriccion**: ninguno tecnico.
- **Transferible**: si. Es el precedente directo de la seccion 17 de la spec
  ("Supervised TF-IDF: TF-IDF x peso discriminativo vs binario vs log-count ratio").
- **Adaptacion (experimento v4)**: `V4-C02` - comparar cinco esquemas de ponderacion sobre
  la misma representacion word(1,2) + char_wb(3,5) de V03: (i) TF-IDF de V03 (baseline),
  (ii) binario, (iii) sublinear, (iv) `x log(1 + df_j / df_pos_j)` (odds-style con
  smoothing), (v) `x w_j(etiqueta)` de `V4-C01`. **Todos los esquemas se calculan dentro
  del pliegue.** Reportar F1 por etiqueta y por pliegue.
  Regla de decision: si ningun esquema supera a V03 en >= 4/5 pliegues y con bootstrap
  pareado, se cierra la familia de ponderacion con `NO_SIGNAL` y no se insiste (spec 83).

### 5.3 [V4-L24] Tang (2024). "A generic multi-level framework for building term-weighting schemes in text classification" (The Computer Journal 2024)

- **DOI**: 10.1093/comjnl/bxae068
- **Problema**: dar un marco de **multiples niveles** para construir esquemas de ponderacion,
  en lugar de proponer un esquema concreto mas, y separar el marco del instancia.
- **Metodo**: marco generico con niveles (frecuencia, distribucion de clase, dispersion) y
  esquemas derivados; validacion cruzada.
- **Restriccion**: ninguno.
- **Transferible**: si, como **plantilla de barrido**: en lugar de 5 esquemas ad hoc, definir
  una rejilla de (nivel de frecuencia) x (senal supervisada) x (suavizado) y ejecutar el
  barrido completo una sola vez. Es la forma correcta de respetar la multiplicidad (spec 72):
  un barrido sistematico con conclusion en una vez, no 20 experimentos sueltos.
- **Adaptacion**: se materializa en el **plan de `30_supervised_weighting_v4.py`**
  (`V4-C01`, `V4-C02`, `V4-C03`), que debe reportar la rejilla completa como una sola
  tabla y cerrar con una unica decision.

### 5.4 [V4-L25] Rathi, Mustafi (2022). "The importance of Term Weighting in semantic understanding of text: A review of techniques" (Multimedia Tools and Applications 2022)

- **DOI**: 10.1007/s11042-022-12538-3
- **Problema**: revision de la familia completa de esquemas de ponderacion (basados en
  frecuencia, supervisados, position-based y semanticos), con su comportamiento empirico.
- **Metodo**: revision bibliografica con taxonomia y comparacion.
- **Restriccion**: incluye esquemas semanticos basados en embeddings (prohibidos); solo se
  usa la parte de frecuencia y supervision.
- **Transferible**: si, como **mapa de cobertura**: permite afirmar que la familia de
  ponderacion esta bien caracterizada y que el subespacio "supervisado por clase con
  suavizado" es el unico hueco razonable.
- **Adaptacion**: sin experimento propio; sustenta el diseno de `V4-C01`-`V4-C03`.

### 5.5 [V4-L26] Robertson (2025). "BM25 and all that - a look back" (SIGIR 2025, tutorial/full paper)

- **DOI**: 10.1145/3726302.3729905
- **Problema**: revisar y actualizar la familia Okapi BM25 y sus variantes, y abandonar
  conjuntamente con la recuperacion de informacion (el campo ha marcado el BM25 como
  "legacy").
- **Metodo**: sintesis historica y tecnica de BM25 y variantes (k1, b, saturacion de
  frecuencia, normalizacion por longitud).
- **Restriccion**: BM25 esta disenado para **rankings** (recuperacion), no para
  clasificacion. Es un ajuste, no una baja.
- **Transferible**: si, y con una advertencia importante. La**forma funcional** de BM25
  (saturacion `tf/(tf+k1)` mas normalizacion por longitud `b`) es una **alternativa a la
  saturacion sublineal de TF-IDF** y a la funcion binaria, con un solo hyperparametro
  interpretable (`k1`) mas un segundo de longitud (`b`). En clasificacion, la evidencia
  empirica de que BM25 supera a TF-IDF es **débil y dependiente del dominio**; por eso se
  propone como variante mas, no como sustitucion.
- **Adaptacion (experimento v4)**: `V4-C03` - mismarepresentacion base, con la funcion de
  termino sustituida por BM25 (k1 en {0.6, 1.2, 2.0} x b en {0.0, 0.3, 0.75}) y el
  `sublinear_tf` desactivado. Barrido de 9 puntos, una sola tabla, decision unica.

### 5.6 [V4-L27] Nota sobre NB y log-odds con smoothing (bloque ya ejecutado)

La peticion menciona "NB re-ponderado" y "log-odds ratio con smoothing". **Este bloque ya
fue ejecutado y cerrado en la fase v2**: `E2-15-NB-WORD` (0.7538), `E2-15-NB-CHAR` (0.7380),
`E2-15-NBSVM` (0.7612), `E2-15-NBSVM-SVC` (0.7429), todas "NO adoptada" frente a V03
(0/5 o 2/5 pliegues favorables). Por la regla `NO_REPEATING` (seccion 84), **esta revision
no propone reintentar el log-count ratio global**. La distincion relevante para v4 es que
los pesos **por etiqueta** de `V4-C01` y el barrido de esquemas de `V4-C02` son
mecanismos genuinamente distintos del log-count ratio global ya cerrado.

---

## 6. Tema 5 - Prediccion multi-etiqueta con dependencia entre etiquetas

### 6.1 [V4-L28] Read, Pfahringer, Holmes, Frank (2021). "Classifier Chains: A Review and Perspectives" (JAIR 2021)

- **DOI**: 10.1613/jair.1.12376 | **Nota de alcance**: 2021, fuera del rango. Se incluye
  por ser la referencia canonica de la seccion 89 de la spec ("clasificadores de cadena").
- **Problema**: sintetizar y poner en perspectiva la familia de Classifier Chains y sus
  variantes, con su dependencia del orden de etiquetas.
- **Metodo**: revision con analisis teorico y empirico de las variantes (cadena, cadena con
  confianza/Probabilidad, cadena sparcia, MCSP/Rama PCS).
- **Restriccion**: ninguna; todos los modelos son lineales/generativos.
- **Transferible**: si, de forma directa. La conclusion util para nosotros: con pocas
  etiquetas y bastante desbalanceo, la **cadena sobre Regresion Logistica** es
  aproximadamente tan buena como cualquier Red neuronal en la practica, y la ganancia
  viene de **modelar la dependencia**, no de aumentar la capacidad del modelo.
- **Adaptacion (experimento v4)**: `V4-E04` - Classifier Chain con 3 etiquetas en 2
  ordenes (HS->TR->AG y HS->AG->TR) sobre los mismos folds de V03, con caracteristicas de
  segunda etapa = decision de la etiqueta anterior (no la etiqueta, para no usar
  informacion de test). Comparar contra la independencia binaria de V03. Nota: el
  `experiments.csv` ya tiene `E2-01/E2-02` "cadenas de etiquetas" en v2; antes de ejecutar
  hay que verificar en `experiments.csv` que la implementacion es genuinamente distinta
  (seccion 84). Si el resultado de v2 ya fue "independencia + post-proceso gana", `V4-E04`
  debe ejecutarse **solo** con la caracteristica de continuacion de decision, no
  replicando la cadena de etiqueta dura.

### 6.2 [V4-L29] Bogatinovski, Todorovski, Dzeroski, Kocev (2022). "Comprehensive comparative study of multi-label classification methods" (Expert Systems with Applications 2022)

- **DOI**: 10.1016/j.eswa.2022.117215
- **Problema**: comparacion **exhaustiva** de familias de metodos multi-etiqueta
  (Binary Relevance, Classifier Chains, Label Powerset, Algoritmos de descomposicion,
  ensembles) sobre su capacidad de explotar la dependencia entre etiquetas.
- **Metodo**: revision experimental con decenas de metodos y conjuntos de datos,
  evaluando clasificacion, ranking y complejidad computacional.
- **Restriccion**: ningun modelo prohibido; son metodos de aprendizaje automatico general.
- **Transferible**: si. Es la **referencia de decision** para la seccion 35 de la spec
  ("Joint label model: potencias + multinomial"). Su conclusion tipica y bien reproducida
  es que: (i) la dependencia entre etiquetas **existe** y mejora con ella;
  (ii) **Label Powerset** gana cuando el numero de combinaciones activas es pequeno; con
  pocas etiquetas, ese es nuestro caso.
- **Adaptacion (experimento v4)**: `V4-E01` es exactamente **Label Powerset
  materializado**: multinomial sobre las combinaciones observadas en TRAIN
  {000, 100, 101, 111} (4 clases) con `LogisticRegression(class_weight='balanced')` sobre
  la misma matriz dispersa word(1,2)+char_wb(3,5) de V03, marginalizado a tres columnas
  con los umbrales OOF preexistentes. Y `V4-E03` es la version **jerarquica**:
  P(HS) binario, y luego P(TR|HS=1), P(AG|HS=1,TR) condicionales. Este trio
  (independencia / powerset / jerarquico) es el experimento central del track E y debe
  ejecutarse en **un solo script con las tres filas y una unica decision** (P3).

### 6.3 [V4-L30] Garcia-Pedrajas, Cuevas-Munoz, Cerruela Garcia, de Haro-Garcia (2024). "A thorough experimental comparison of multilabel methods for classification performance" (Pattern Recognition 2024)

- **DOI**: 10.1016/j.patcog.2024.110342
- **Problema**: comparacion experimental recente y amplia de metodos multi-etiqueta, con
  enfasis en el comportamiento en condiciones realistas (alta cardinalidad, desbalanceo).
- **Metodo**: estudio experimental sobre multiples familias de metodos, con analisis de
  sensibilidad y coste.
- **Restriccion**: ninguna.
- **Transferible**: si, como **segunda fuente independiente** de la misma conclusion. Dos
  comparativas recientes y de alto impacto que coinciden dan confianza al hallazgo
  "powerset con pocas combinaciones activas".
- **Adaptacion**: refuerza `V4-E01`; no aporta experimento propio.

### 6.4 [V4-L31] Huang, Hu, Lu, Fan, Xu (2024). "Application of Label Correlation in Multi-Label Classification: A Survey" (Applied Sciences 2024)

- **DOI**: 10.3390/app14199034
- **Problema**: revisar los usos de la **correlacion entre etiquetas** en clasificacion
  multi-etiqueta: particionado, seleccion de caracteristicas, seleccion de ejemplos,
  imputacion.
- **Metodo**: revision taxonomica.
- **Restriccion**: ninguna.
- **Transferible**: si. Es la referencia que justifica usar la dependencia
  HS->TR como **una sola fuente de senal** (no tres problemas independientes) y que
  documenta la seleccion de caracteristicas por etiqueta como uso establecido.
- **Adaptacion**: sustenta `V4-C01` (features por etiqueta) y `V4-E01`.

### 6.5 [V4-L32] Chalkidis, Sogaard (2022). "Improved Multi-label Classification under Temporal Concept Drift: Rethinking Group-Robust Algorithms in a Label-Wise Setting" (Findings ACL 2022)

- **DOI**: 10.18653/v1/2022.findings-acl.192
- **Problema**: en documentos con cientos de clases, infrequentisimas y con deriva
  temporal, el remuestreo uniforme no es la mejor respuesta; el trabajo reencuadra los
  **algoritmos group-robust** (IRM, Spectral Decoupling) como algoritmos de adaptacion.
- **Metodo**: comparacion de algoritmos de optimizacion group-robust frente a remuestreo
  basado en una distribucion objetivo conocida.
- **Resultado reportado**: IRM y Spectral Decoupling superan a los enfoques de remuestreo
  para imbalance y drift, y son **mucho mejores en clases minoritarias**; el efecto crece
  con el tamaño del conjunto de etiquetas.
- **Restriccion**: IRM y Spectral Decoupling son metodos de optimizacion neuronal
  (prohibidos). **Pero la conclusion es independiente del metodo**: con conjunto de
  etiquetas pequeño (3), el argumento "mejorar el regimen de optimizacion" **no aplica**,
  y la evidencia de que el efecto crece con el numero de etiquetas nos dice que aqui el
  problema es de otro tipo.
- **Transferible**: si, con una **implicacion negativa util**: no intentar reproducir IRM
  ni Spectral Decoupling (prohibido y ademas inescalable a 3 etiquetas). Lo que si queda
  es el diagnostico: la clases minoritaria de nuestro corpus (TR, 29.98 %) se beneficia mas
  de una funcion de perdida/pesos bien elegida que de un algoritmo de optimizacion
  exotico, lo cual apoya `V4-B06` (barria de `class_weight` por etiqueta).
- **Adaptacion**: justifica `V4-B06` y cierra (como "no aplicable") cualquier via de
  optimizacion robusta de la seccion 14 para el track B.

### 6.6 [V4-L33] Chen, Li, Xi, Yu, Xiong (2023). "Rare Codes Count: Mining Inter-code Relations for Long-tail Clinical Text Classification" (ClinicalNLP 2023)

- **DOI**: 10.18653/v1/2023.clinicalnlp-1.43
- **Problema**: enorme brecha de rendimiento entre codigos raros y frecuentes en
  clasificacion multi-etiqueta clinica; el modelo no "excava" las relaciones entre
  etiquetas.
- **Metodo**: codificador de codigos mejorado con relaciones entre codigos, partiendo de
  **descripciones simples** de los codigos (sin conocimiento externo pesado); mejora
  explicitamente el rendimiento en codigos raros.
- **Restriccion**: el dataset es clinico y externo (prohibido). El metodo es neuronal.
- **Transferible**: si, por el principio que valida: **las relaciones entre etiquetas
 pueden explotarse para mejorar las combinaciones raras sin conocimiento externo**, solo
  con lo que ya esta en los datos. En nuestro caso las combinaciones raras son 110 (n=28)
  y 001 (n=26) (seccion 37 de la spec).
- **Adaptacion (experimento v4)**: `V4-E05` - para las combinaciones **frecuentes** (000,
  100, 101, 111) construir un **modelo por combinacion** (o un bloque de features
  interaccion) que use como entradas las salidas de los tres clasificadores binarios,
  y usar la prediccion de combinacion como **prior** para estabilizar las combinaciones
  raras en `V4-E01`. Concretamente: reportar F1 por combinacion antes y despues de
  aplicar el prior. Con 28 y 26 ejemplos, la expectativa realista es baja; si la mejora es
  nula, la seccion 37 se cierra con `NO_SIGNAL` documentado.

---

## 7. Tema 6 - Representaciones sparse y subword sin transformers

### 7.1 [V4-L34] Bates, Gurevych (2024). "Like a Good Nearest Neighbor: Practical Content Moderation and Text Classification" (EACL 2024)

- **DOI**: 10.18653/v1/2024.eacl-long.17
- **Problema**: los sistemas de clasificacion de few-shot dependen de promptings y modelos
  de miles de millones de parametros, insostenibles para moderacion de contenido a escala.
- **Metodo**: LaGoNN, una modificacion de SetFit que **no introduce parametros
  aprendibles**: altera el texto de entrada con informacion de su **vecino mas cercano**
  en TRAIN (su etiqueta y su texto), de modo que el dato nuevo parezca una instancia sobre
  la que el modelo fue optimizado. Se evalua en moderacion de contenido bajo **cuatro
  distribuciones de etiqueta** y en clasificacion general y multilingue.
- **Restriccion**: el punto de partida (SetFit) es un encoder de frase preentrenado
  (prohibido). Pero el **mecanismo** -inyectar informacion del vecino mas cercano de
  TRAIN- no usa nada preentrenado.
- **Transferible**: si, y es el mejor articulo de esta seccion para nuestras restricciones.
  La version lineal de la idea es: en vez de vectores de texto, usar como **feature del
  clasificador lineal** una **expresion de similitud con los vecinos de TRAIN**
  (coseno de TF-IDF word, Jaccard de char 3-5, coincidencia de la combinacion de etiquetas
  del vecino). Eso es un track F puro (secciones 55-58) y **no necesita ningun encoder**.
- **Adaptacion (experimento v4)**: `V4-F02` - computar, para cada tweet de TRAIN, el
  vecino mas cercano en el resto de TRAIN (excluyendo el propio, dentro del fold) y
  anadir al clasificador lineal un **small bloque de 3-5 features de vecindad** (coseno
  word, coseno char, Jaccard char, etiqueta combinacion del vecino como one-hot de 4
  categorias, y distancia de longitud). Regla critica: el indice de vecinos se construye
  **solo con TRAIN del fold**; en test, el vecino se busca en TRAIN completo (eso si es
  legal, es `transform`).

### 7.2 [V4-L35] Hossain, Kabir (2025). "Context Minimization for Resource-Constrained Text Classification: Optimizing Performance-Efficiency Trade-offs through Linguistic Features" (Findings EMNLP 2025)

- **DOI**: 10.18653/v1/2025.findings-emnlp.833
- **Problema**: reducir el coste de la clasificacion de texto en entornos con recursos
  limitados, sin perder señal.
- **Metodo**: marco guiado por la **linguistica funcional** (estructura tema-rema) que
  integra features **posicionales, sintacticas, semanticas y estadisticas** para identificar
  configuraciones de bajo contexto; protocolo iterativo de exploracion de features sobre 6
  benchmarks, analisis con SHAP y tests de significancia. Gananzas de eficiencia de
  69-75 % de memoria y 81-87 % de tiempo con reduccion de macro-F1 de solo 1.39-3.10 %.
- **Restriccion**: el marco opera sobre PLM; la parte semantica requiere embeddings
  (prohibida). La parte **posicional/sintactica/estadistica** y la **metodologia de
  exploracion iterativa de features** son legales.
- **Transferible**: si, por la metodologia: (a) la **reduccion de contexto con metrica
  funcion de palabras** es una idea sparse compatible y distinta de lo ya hecho (el canal
  de palabras funcionales se uso en HatEval, pero no como reduccion de contexto sobre el
  canal word completo); (b) el uso de **SHAP sobre un clasificador lineal** identifica que
  familias de features sostienen el F1 y evita la acumulacion ciega (que fue el
  error de Nina-Alcocer, R05 en la revision previa).
- **Adaptacion (experimento v4)**: `V4-D01` - canal nuevo: **contenido + ventanas de
  palabras funcionales**, es decir, TF-IDF sobre una version del tweet en la que las
  palabras de contenido (las de mayor TF-IDF) se sustituyen por su clase, y se añade como
  canal separado (no mezclado, por la regla 86 de la spec: POS/patrones como sparse
  explicito, no como bloque denso). Second: `V4-D02` - SHAP sobre el LR de V03 para
  **ranking de familias de features** y tabla de densidad. Es un diagnostico, no un modelo.

---

## 8. Tema 7 - Perplejidad o verosimilitud class-conditional como features

**Resultado: sin evidencia reciente transferible (2022-2026).**

Se realizaron busquedas explicitas en Crossref (todo el corpus), OpenAlex
(`title_and_abstract.search`) y arXiv con las consultas "perplexity as a feature text
classification language model score", "class conditional n-gram likelihood features abuse
detection", "text classification generative model likelihood features", "unsupervised text
hardness outlier difficulty". Los unicos trabajos que aparecen con verosimilitud como
caracteristica se dispositifs para **deteccion de texto generado por IA**
(perplejidad de modelos Small Language Model, GLTR), que es un problema de atribucion de
autoria, no de clasificacion de contenido, y ademas usa modelos preentrenados.

**Conclusion operativa**: la seccion 59-60 de la spec (LM estadistico desde cero,
P(text|clase); perplejidad class-conditional) **no tiene respaldo bibliografico reciente
que la respalde**. Esto no significa que sea mala idea, sino que es una **hipotesis
propia sin literatura que la valide** y debe ejecutarse con la expectativa correspondiente
(y con la regla de que, sin mecanismo explicito, un resultado nulo se cierra con
`NO_SIGNAL` documentado, no se reintenta).

- **Adaptacion propuesta pese al vacio**: `V4-G01` - LM de clase desde cero, sin preentrenamiento:
  unigramas y bigramas de **caracteres** y de **palabras** por etiqueta (`P(w|HS=1)`,
  `P(w|HS=0)`, etc.) estimated solo con TRAIN y con **smoothing de Laplace/Jeffreys**; luego
  features por ejemplo: `logP(tweet|HS=1)`, `logP(tweet|HS=0)`, su **margen**, y lo mismo
  para TR y AG; normalizados por longitud. Estos 9 numeros entran como columna densa en
  el meta-aprendiz (no en el canal sparse, para no romper la geometria de V03). Decision
  previa al experimento: se acepta **solo** si mejora en >= 4/5 pliegues **y** aporta
  complementariedad (error no correlacionado con el de V03). Si no, `NO_SIGNAL`.

### 8.1 [V4-L36] Nota sobre la ausencia de "NBSVM" reciente

Tampoco se ha encontrado una reevaluacion 2022-2026 revisada por pares del esquema
NBSVM (log-count ratio de Wang & Manning) con rigor suficiente. Los unicos trabajos
recuperados ("Sentiment Analysis Based on Aspects Using FastText Feature Expansion and
NBSVM Classification Method", JoSYC 2022; "Vaccine sentiment analysis using BERT + NBSVM
and geo-spatial approaches", J. Supercomputing 2023) son de baja calidad metodologica,
no usan el corpus de odio, y ademas NBSVM puro ya esta ejecutado y cerrado (`E2-15`).
Se registra como vacio.

---

## 9. Tema 8 - Discurso de odio en espanol reciente

### 9.1 [V4-L37] Castillo-Lopez, Riabi, Seddah (2023). "Analyzing Zero-Shot transfer Scenarios across Spanish variants for Hate Speech Detection" (VarDial 2023)

- **DOI**: 10.18653/v1/2023.vardial-1.1
- **Problema**: las variaciones entre **geolectos** (espanol latinoamericano frente a
  espanol europeo) siguen siendo un problema para los modelos que dependen de informacion
  **lexical latente**, y el aspecto cultural de la deteccion de odio se pasa por alto.
- **Metodo**: analisis del rendimiento de modelos de deteccion de odio sobre distintas
  variantes del espanol, incluyendo un **nuevo conjunto de datos de discurso de odio hacia
  inmigrantes en Twitter** construido para cubrir esas variantes; se prueba mBERT y BETO
  como base de transferencia.
- **Resultados reportados**: los modelos entrenados para una variante de espanol se ven
  afectados cuando no se consideran las otras variaciones: las expresiones de odio varian
  de region a region dentro de una misma lengua.
- **Restriccion**: los modelos son preentrenados (prohibidos) y el dataset es externo
  (prohibido). El dataset no se usa.
- **Transferible**: si, y con una **implicacion metodologica concreta** que ya tenemos
  instrumentada: nuestro corpus es un unico bloque (no podemos separar variantes), pero
  **si podemos medir la heterogeneidad regional dentro del corpus** mediante marcas
  ortograficas y lexicales que ya existen en el propio texto (voseo, "vos", "che",
  uso de "vosotros", "computadora", "celular", "chaval", "malparido" frente a "cabron",
  "hijoputa" con escritura particular, etc.), **todas derivadas de TRAIN y documentadas
  (D-007)**.
- **Adaptacion (experimento v4)**: `V4-F03` - **diagnostico no supervisado de shift**
  train<->test** (secciones 52-53 de la spec): proyectar TRAIN y TEST con el mismo
  vectorizador (aprendido en TRAIN) en un espacio de baja dimension con `TruncatedSVD` de
  las char n-grams, y comparar las distribuciones marginales y la distancia al centroide
  de TRAIN, **etiqueta por etiqueta** para identificar que etiquetas concentran el
  desplazamiento. No se usa ninguna etiqueta de test (§38 NO_TEST_FITTING). Entregable:
  informe de shift por etiqueta; decide si `V4-D01` (canal funcional) tiene sentido.

### 9.2 [V4-L38] Perez, Miguel, Cotik (2025). "Exploring Large Language Models for Hate Speech Detection in Rioplatense Spanish" (Findings NAACL 2025)

- **DOI**: 10.18653/v1/2025.findings-naacl.400
- **Problema**: medir el rendimiento de LLM (ChatGPT 3.5, Mixtral, Aya) en deteccion de
  discurso de odio en espanol rioplatense, con razonamiento chain-of-thought, frente a un
  clasificador BERT afinado.
- **Resultados reportados**: los LLM muestran **menor precision** que el clasificador BERT
  afinado; aun asi son sensibles a casos muy matizados, en particular discurso de odio
  **homofobico y transfobico**, donde a veces detectan insultos o coloquialismos de
  dificil acceso.
- **Restriccion**: los modelos son LLM (prohibidos); no se usan.
- **Transferible**: si, por un hallazgo de **error esperable** muy util: la clase de
  hardest del dominio (odio homofobico/transfobico) es justamente donde la señal es
  **sublexical y ortografica** (insultos enmascarados, escritura fonetica). Eso conecta
  directamente con el canal ortografico que ya se probo en `E2-10-ORT-*` y sugiere que
  **la combinacion char_wb(3,5) + ortografico es la unica defensa realista** en esa clase
  de error, y que el techo de V03 en TR/HS tiene esa naturaleza concreta.
- **Adaptacion**: entra en `reports/error_taxonomy_v4.md` como caracterizacion de la clase
  de error residual mas dificil. No genera experimento propio (el canal ortografico ya
  fue barrido y cerrado en v2).

### 9.3 [V4-L39] Abercrombie, Hovy, Prabhakaran (2023). "Temporal and Second Language Influence on Intra-Annotator Agreement and Stability in Hate Speech Labelling" (LAW-XVII 2023)

- **DOI**: 10.18653/v1/2023.law-1.10
- **Problema**: la consistencia del etiquetado de discurso de odio se ve afectada por
  factores temporales y por el hecho de que el anotador no es su lengua materna.
- **Metodo**: estudio de anotacion que mide el acuerdo intra-anotador y su estabilidad a lo
  largo del tiempo, manipulando el tiempo transcurrido y la condicion de L2.
- **Restriccion**: requiere un corpus con anotaciones repetidas en el tiempo (no lo tenemos).
- **Transferible**: si, por la **implicacion**: si la acuerdo intra-anotador deciae, entonces
  una parte de las etiquetas deHatEval es **irreducible**, y la mejor estrategia no es
  "limpiar" sino **dejar de exigir precision por debajo de ese nivel**. Es un argumento
  adicional (junto a Plank 2022 y Davidson 2019) a favor de §72 MULTIPLICIDAD.
- **Adaptacion**: sin experimento. Refuerza la redaccion honesta del informe final.

### 9.4 [V4-L40] Ron, Levi, Oshri, Shenhav (2023). "Factoring Hate Speech: A New Annotation Framework to Study Hate Speech in Social Media" (WOAH 2023)

- **DOI**: 10.18653/v1/2023.woah-1.21
- **Problema**: el esquema "categorico" de discurso de odio mezcla en una sola etiqueta
  varias dimensiones ortogonales (target, contenido, forma), lo que impide medir
  correctamente las relaciones entre ellas.
- **Metodo**: marco de anotacion que **factura** el discurso de odio en componentes y
 Zkouseni con ello la estructura de dependencias.
- **Restriccion**: el marco es para construir un corpus nuevo (prohibido). No se usan datos.
- **Transferible**: si, como **justificacion del track E**. Es el respaldo de que la
  estructura HS -> (TR, AG) de HatEval no es un accidente del anotador sino una propiedad
  del concepto, y por tanto que modelarla (en vez de ignorarla) es lo correcto.
- **Adaptacion**: sustenta `V4-E01`, `V4-E03` y `V4-E05`. Sin experimento propio.

### 9.5 [V4-L41] Zhou, Karamolegkou, Chen, Hershcovich (2023). "Cultural Compass: Predicting Transfer Learning Success in Offensive Language Detection with Cultural Features" (Findings EMNLP 2023)

- **DOI**: 10.18653/v1/2023.findings-emnlp.845
- **Problema**: las tareas subjetivas como la deteccion de lenguaje ofensivo dependen
  fuertemente de valores culturales; se investiga si se puede **predecir** el exito de una
  transferencia entre culturas a partir de features culturales.
- **Metodo**: se usan encuestas de valores culturales y **distancia entre palabras
  ofensivas** entre culturas como predictores del exito de transferencia.
- **Restriccion**: requiere datos de encuestas culturales externas (prohibido) y la
  distancia entre palabras ofensivas requiere lexicos externos (prohibido). El metodo es
  transfer learning (prohibido).
- **Transferible**: **no directamente**. Se cita unicamente como evidencia de que la
  dependencia de la tarea de la variedad linguistica es un phenomenon documentado en
  2023-2025, y por tanto de que la hipotesis de shift de `V4-F03` es plausible.
- **Adaptacion**: sin experimento propio. **Es el unico trabajo de la lista cuya idea no es
  transferible bajo las restricciones**; se mantiene por su valor de contexto.

---

## 10. Temas con cobertura insuficiente (declaracion explicita)

| Tema solicitado | Estado | Detalle |
|---|---|---|
| **7. Perplejidad / verosimilitud class-conditional como features** | **SIN EVIDENCIA RECIENTE TRANSFERIBLE** | Ningun trabajo revisado por pares 2022-2026 usa la verosimilitud de un LM como feature para clasificacion de contenido en redes sociales. Los trabajos de perplejidad son de deteccion de texto generado. La idea queda como hipotesis propia sin respaldo (`V4-G01`). |
| **"Companion labels"** (seccion 2 de la peticion) | **SIN EVIDENCIA VERIFICABLE** | No se ha podido confirmar ningun trabajo revisado por pares 2022-2026 que use el termino en el sentido de mecanismo de estimacion de ruido. No se cita ninguna referencia no confirmada. |
| **NBSVM / log-count ratio re-evaluado (2022-2026)** | **SIN EVIDENCIA SOLIDA** | Solo trabajos de baja calidad metodologica y de otros dominios. Ademas la idea ya fue ejecutada y cerrada en v2 (`E2-15`). |
| **Word-shape explicito como familia de features (2022-2026)** | **COBERTURA DEBIL** | No se ha encontrado un trabajo 2022-2026 revisado por pares que valide word-shape (patrones de mayusculas/digitos/puntuacion) como canal separado para deteccion de odio en espanol. El trabajo mas cercano (2025.findings-emnlp.833) es sobre reduccion de contexto y no sobre word-shape. La feature de forma ya esta cubierta de facto por char n-grams en v2. |
| **Representaciones subword "estilo wordpiece" sin transformers (2022-2026)** | **COBERTURA DEBIL** | El trabajo reciente sobre subword asume un encoder preentrenado. La unica aproximacion legal es la char n-grams, ya implementada en v2. Se cierra como linea. |
| **"Mentored training" con pesos (seccion 2 de la peticion)** | **SIN TRABAJO IDENTIFICADO** | Se han encontrado variantes de self-training ponderado en texto (p.ej. "SAT: Improving Semi-Supervised Text Classification with Simple Instance-Adaptive Self-Training", Findings EMNLP 2022, 10.18653/v1/2022.findings-emnlp.456) pero **todas dependen de pseudo-etiquetado, prohibido por NO_PSEUDO_LABELING**. Ninguna es transferible bajo las restricciones. Se deja constancia del titulo y DOI para trazabilidad, pero no se incluye como referencia del informe. |

---

## 11. Top 8 ideas transferibles priorizadas

Prioridad = (valor esperado) x (fidelidad a las restricciones) x (hueco real medido en
`experiments.csv`). Cada implicacion es concreta para `KAGGLE_TOXIC_2026_PUJ`.

### 1. Confident learning por etiqueta con predicciones OOF (Track A, `V4-A03`)
**Trabajos**: Northcutt, Jiang, Chuang (JAIR 2021); Xu, Jing, Yu (Findings EMNLP 2024).
**Implicacion concreta**: aplicar `s(x) = p_hat(y_obs)` **por etiqueta y solo con OOF** a los
1.731 ejemplos de TRAIN. Cruzar `s` con las tres categorias ya calculadas en `V4-B02`
(Facil / Dificil coherente / Posible ruido). Criterio de cierre: si la distribucion de `s`
se solapa entre "Dificil coherente" (296 ejemplos, 20 %) y "Posible ruido" (43 ejemplos,
2.9 %), **el track de ruido se cierra con evidencia y todo el esfuerzo pasa al track de
dureza**. Esta es la idea de mayor relacion valor/coste de toda la revision: cuesta un
script y decide si dos tracks enteros valen la pena.

### 2. Re-declarar el track B desde "ruido" hacia "dureza" (Track B, `V4-B03`)
**Trabajos**: Wu et al. (IEEE TMM 2023, DiscrimLoss); V4-B02 (medicion propia).
**Implicacion concreta**: `V4-B02` ya mostro que los "posibles ruidosos" (2.9 %) tienen
**menor** error medio (0.054) que los "dificiles coherentes" (20.0 %, error 0.351). La
discriminacion de DiscrimLoss dice: reponderar la **dureza**, no el ruido. Por tanto
`V4-B03` debe usar `sample_weight` derivado de `error_score` y `bce_score` OOF
(seccion 11 de la spec), con topes (ningun ejemplo baja de peso 0.3 y ninguno sube de
2.0) y **sin borrar nada** (seccion 8). Techo realista: 20 % de los ejemplos, no 3 %.

### 3. Ponderacion de features **por etiqueta** (Track C, `V4-C01`)
**Trabajos**: Attieh, Tekli (KBS 2023); Huang et al. (Applied Sciences 2024); Tang
(Computer Journal 2024).
**Implicacion concreta**: tres vectores de pesos `w_HS`, `w_TR`, `w_AG` (chi2 con
smoothing, information gain, MI por histograma de conteos) calculados **dentro de cada
pliegue**, aplicados como `TFIDF_j * w_etiqueta_j` con **un clasificador por etiqueta**.
Es **genuinamente distinto** de `E2-15-NBSVM` (que uso un unico log-count ratio global y
cerro en -0.016). Es la idea con mejor soporte bibliografico relativo de toda la revision.

### 4. Barrido sistematico de esquemas de ponderacion en una sola vez (Track C, `V4-C02` + `V4-C03`)
**Trabajos**: Tang, Li, Li (ESWA 2022); Robertson (SIGIR 2025); Rathi, Mustafi (MTAP 2022).
**Implicacion concreta**: una **unica tabla** con (i) TF-IDF V03, (ii) binario, (iii)
sublinear, (iv) odds-style `log(1 + df_j/df_pos_j)`, (v) ponderado por etiqueta de
`V4-C01`, y (vi) BM25 con `k1` en {0.6, 1.2, 2.0} x `b` en {0.0, 0.3, 0.75}. Todo con los
mismos folds, todo en `30_supervised_weighting_v4.py`, y **una sola decision** al final.
Regla §72: si ninguna variante gana en >= 4/5 pliegues con bootstrap pareado favorable, la
familia entera se cierra como `NO_SIGNAL` y no se reintenta.

### 5. Modelo conjunto sobre el subespacio factible de etiquetas (Track E, `V4-E01` + `V4-E03`)
**Trabajos**: Bogatinovski et al. (ESWA 2022); Garcia-Pedrajas et al. (Pattern Recognition
2024); Ron et al. (WOAH 2023); Cercas Curry et al. (WOAH 2024); Weerasooriya et al.
(Findings ACL 2023).
**Implicacion concreta**: multinomial de **4 clases** sobre {000, 100, 101, 111} (las unicas
observadas; TR solo existe con HS=1) marginalizado a tres columnas, comparado en un mismo
script contra (a) la independencia binaria de V03 y (b) la version jerarquica
`P(HS) x P(TR|HS=1) x P(AG|HS=1,TR)` con las **probabilidades condicionales empíricas de
TRAIN como prior suave** (no reglas duras). Es la seccion 89 ("joint structured output y
soft priors, no solo cadenas") y el unico track con dos comparativas recientes de alto
impacto a favor.

### 6. Barrido de cost-sensitive learning **por etiqueta** (Track B, `V4-B06`)
**Trabajos**: Zhang, Hangya, Fraser (WOAH 2024); Chalkidis, Sogaard (Findings ACL 2022).
**Implicacion concreta**: `class_weight` en {None, balanced, {1:a,0:1} con a en {0.7,0.8,0.9,
1.1,1.25}} por etiqueta, mas sobremuestreo y submuestreo dentro del pliegue. El gap esta
declarado en la propia spec (seccion 42: "`class_weight='balanced'` nunca se barrio por
etiqueta") y ahora tiene respaldo externo explicito: el estudio de WOAH 2024 encuentra que
focal loss y sobremuestreo son los mas versatiles, y que sobremuestreo/entropia ponderada
funcionan mejor en el caso binario moderado, que es exactamente el nuestro. Nota: focal
loss no aplica a `LogisticRegression` de sklearn; traducirla a `sample_weight` por ejemplo.

### 7. Features de vecindad no supervisadas dentro del modelo lineal (Tracks F + B, `V4-F02`, `V4-F01`)
**Trabajos**: Bates, Gurevych (EACL 2024, LaGoNN); Cordeiro, Carneiro (Pattern Recognition
2025, ANNE).
**Implicacion concreta**: anadir al clasificador lineal un bloque de 3-5 features de
vecindad calculadas contra TRAIN (coseno TF-IDF word, Jaccard char 3-5, combinacion de
etiquetas del vecino, diferencia de longitud), con el indice construido **solo con TRAIN
del fold** y, en test, buscado en TRAIN completo. Es la adaptacion lineal de LaGoNN (que
inyecta informacion del vecino mas cercano) y de ANNE (dos vistas: modelo y datos). Ataque
directo a las secciones 55-58 y a la linea de vecindad, que nunca se ha ejecutado.

### 8. Descomposicion funcional de los errores por tipo de caso (Track D, `V4-D05`, `V4-D03`)
**Trabajos**: de Araujo, Roth (NLP Power! 2022); Perez, Miguel, Cotik (Findings NAACL 2025).
**Implicacion concreta**: particionar el OOF congelado de V03 en subconjuntos definidos
**solo con features no supervisadas de TRAIN** (negacion, segunda persona, target
colectivo, atribucion/citacion, emoji/onomatopeya, longitud, ortografia fonetica) y
reportar densidad de error y F1 por subconjunto y por celda de la matriz de confusion.
Es la version barata de HateCheck que las restricciones permiten, y responde al hallazgo de
Perez et al. de que la clase residual mas dificil (odio homofobico/transfobico) es
precisamente la sublexical. Sirve para decidir si `V4-D01` (canal de palabras funcionales)
tiene fundamento y para cerrar honestamente la seccion 7 (HS implicito).

---

## 12. Mapa experimento v4 -> fuente bibliografica

| Experimento | Script previsto | Track | Fuentes que lo justifican |
|---|---|---|---|
| `V4-A03` | `27_label_consistency_v4.py` | A | Northcutt 2021; Xu, Jing, Yu 2024 |
| `V4-A04` | `27_label_consistency_v4.py` | A | Wang, Plank 2023; Ivey, Gauch, Jurgens 2025 |
| `V4-B03` | `29_reweighting_v4.py` | B | Wu et al. 2023; Xu, Jing, Yu 2024; DiscrimLoss |
| `V4-B04` | `29_reweighting_v4.py` | B | Xu et al. 2024 (par instancia-etiqueta) |
| `V4-B06` | `29_reweighting_v4.py` | B | Zhang, Hangya, Fraser 2024; Chalkidis, Sogaard 2022 |
| `V4-B07` | `29_reweighting_v4.py` | B | Huang et al. 2024 (LRSL) |
| `V4-C01` | `30_supervised_weighting_v4.py` | C | Attieh, Tekli 2023; Huang et al. 2024; Tang 2024 |
| `V4-C02` | `30_supervised_weighting_v4.py` | C | Tang, Li, Li 2022; Rathi, Mustafi 2022 |
| `V4-C03` | `30_supervised_weighting_v4.py` | C | Robertson 2025 |
| `V4-D01` | `34_structural_sparse_v4.py` | D | Hossain, Kabir 2025 |
| `V4-D05` | `34_structural_sparse_v4.py` | D | de Araujo, Roth 2022; Perez, Miguel, Cotik 2025 |
| `V4-E01` | `31_joint_label_v4.py` | E | Bogatinovski et al. 2022; Garcia-Pedrajas et al. 2024; Ron et al. 2023 |
| `V4-E02` | `31_joint_label_v4.py` | E | Fleisig, Abebe, Klein 2023 |
| `V4-E03` | `31_joint_label_v4.py` | E | Weerasooriya et al. 2023; Cercas Curry et al. 2024 |
| `V4-E05` | `31_joint_label_v4.py` | E | Chen et al. 2023 |
| `V4-F01` | `35_discovery_v4.py` | F | Cordeiro, Carneiro 2025 |
| `V4-F02` | `35_discovery_v4.py` | F | Bates, Gurevych 2024 |
| `V4-F03` | `35_discovery_v4.py` | F | Castillo-Lopez et al. 2023; Zhou et al. 2023 |
| `V4-G01` | `33_class_conditional_lm_v4.py` | G | **sin respaldo bibliografico** (ver seccion 8) |

---

## 13. Limitaciones de esta revision

1. **Cobertura de bases**: se usaron las APIs de Crossref (todo el corpus, con filtro de
   prefijo para ACL Anthology), OpenAlex y arXiv. **DBLP y Semantic Scholar noILDRaron**
   estar disponibles en esta sesion (rate limiting persistente), lo que reduce la cobertura
   de articulos de conference no indexados en Crossref (en particular proceedings de IEEE
   y de ACM con DOI de conferences, no journals). Las comparativas multi-etiqueta se
  localizaron por esa via alternativa (OpenAlex) y quedaron verificadas.
2. **ACL Anthology**: el buscador nativo usa Google Custom Search (renderizado por
   JavaScript), por lo que **no se pudo hacer full-text search sobre PDFs del Anthology**;
   la busqueda fue sobre metadatos Crossref y abstracts. Es posible que exista literatura
   relevante en WOAH, LAW o Insights que no aparezca en metadatos.
3. **Sin lectura completa de PDFs**: para los trabajos cuya restricción es evidente por el
   modelo usado (neural, PLM, embeddings), la ficha se apoya en el abstract. Para los
   trabajos de los que depende una decision concreta (`V4-B06`, `V4-C01`) el abstract es
   suficiente para la **motivacion** pero la **magnitud** del efecto deberia confirmarse
   leyendo el PDF antes de escribir el script.
4. **Sesgo de publicacion**: los estudios de barrido de metodos (Bogatinovski 2022,
   Garcia-Pedrajas 2024, Zhang/Hangya/Fraser 2024) tienen sesgo hacia "el metodo propuesto
   gana". Las conclusiones se han usado solo en su parte **taxonomica y de diseno**,
   nunca como numero de mejora.
5. **Solapes con la revision previa**: los trabajos 2016-2021 de `estado_arte.md` (HateCheck,
   HatEval, Wiegand, MineriaUNAM, Davidson) **no se repiten aqui**. Se citan solo cuando
   un trabajo 2022-2026 los usa como base teorica (HateCheck en de Araujo & Roth 2022).
6. **Preprints**: excluido `AlleNoise` (arXiv 2024, Rączkowska et al.) por no tener
   publicacion revisada por pares confirmada; y quedan fuera por la misma razon otras
   candidatos que la busqueda devolvio en arXiv.
7. **HTTP 403 no significa DOI invalido**: 35 de las 38 URLs del CSV respondieron 200 en una
   comprobacion automatica del 2026-09-25. Las 3 restantes
   (`10.1093/comjnl/bxae068`, `10.1145/3726302.3729905`, `10.3390/app14199034`) devolvieron
   403 al seguir la redireccion del DOI, que es un bloqueo anti-bot del editor (Oxford
   University Press, ACM y MDPI respectivamente). Los tres DOIs se resolvieron con titulo,
   autores, ano, revista y tipo de documento correctos contra la API de Crossref, y el
   registro ACM confirma ademas que Robertson 2025 es un articulo completo de los
   proceedings del SIGIR 48.

---

## 14. Reproducibilidad de esta revision

Todos los DOI listados fueron resueltos contra `https://api.crossref.org/works/{doi}` o
`https://api.openalex.org/works?filter=doi:{doi}` en la sesion del 2026-09-25, y todas las
referencias `10.18653/*` fueron confirmadas adicionalmente contra el `.bib` oficial de
ACL Anthology. Los identificadores ACL son estables y apuntan a la pagina del trabajo.
Ningun resultado numerico de esta revision procede de memoria: cuando se citan numeros,
proceden del abstract o del texto del trabajo, y se marcan como "reportado".
