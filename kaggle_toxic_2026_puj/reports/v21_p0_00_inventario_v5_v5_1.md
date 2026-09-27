# V21 P0: inventario y auditoria de V5 y V5.1

Fase V21, P0. Objetivo del encargo: antes de ejecutar nada, auditar exactamente que se
hizo en V5 y V5.1 con los embeddings entrenados desde cero, y construir la tabla de
inventario. Nada se ejecuta hasta tenerla.

Fuente: lectura directa del codigo, no de resumenes previos. Los 23 scripts
`experiments/embedding_*.py`, las filas `EBD-*`, `E-CORP-*` y `E2-15-*` de
`experiments.csv`, `reports/embedding_architecture_v5_1.md`,
`docs/embedding_from_scratch_v5_spec.md` y `docs/competition_rules.md`.

---

## 1. Reglas vigentes (seccion 1 del encargo)

Verificadas contra el texto capturado del organizador
(`docs/competition_rules.md`, 2026-09-22).

Se permite: NLTK, Scikit-learn, TensorFlow, PyTorch, Pandas, Numpy. El dataset de
prueba **no** se usa para entrenamiento. La restriccion adicional del profesor es la
vinculante:

> No se pueden usar transformers ni modelos LLM para la clasificacion del texto; no se
> pueden usar modelos como RoBERTa o BERT. Se pueden usar modelos de vectorizacion,
> pero deben entrenarse desde cero y hacer todo desde cero.

Interpretacion vigente que la fase hereda (D-030, ya resuelta en v5):

| permitido | prohibido |
|---|---|
| corpus externos **no etiquetados** | datasets externos etiquetados |
| embeddings entrenados desde cero por el equipo | embeddings preentrenados, cualquier variante |
| texto de train | test para entrenar |
| | transformers, LLM, RoBERTa, BERT |
| | pseudo-labeling, transduccion no autorizada |

**Ambiguidad que la fase NO resuelve y que se asume en el sentido mas restrictivo**:
las reglas no dicen explicitamente si una lexicon externa de insultos en espanol
cuenta como "modelo de vectorizacion entrenado desde cero" o como recurso etiquetado
externo. No se usara hasta que haya evidencia del profesor.

---

## 2. La tabla de inventario (seccion 2 del encargo)

Todas las filas comparten el bloque disperso de V03 (27.431 TF-IDF: word(1,2) +
char_wb(3,5), `min_df=2`, `sublinear_tf`), el receptor
`LogisticRegression(C=2.0, class_weight="balanced", liblinear)` y **los umbrales
congelados de V03** (0.44 / 0.51 / 0.46). `corrected_cv` de V03 = 0.7772114525165721.
La metrica de cada fila es el `corrected_cv v2` de la fusion. `emb_solo` es el bloque
denso por si solo, con los mismos umbrales.

| Experimento | Corpus | Limpieza | Tokenizacion | Arquitectura | Dim | Pooling | emb_solo | fusion_cv | delta | LB | Que quedo sin probar |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|
| V5 baseline | wikipedia_es 10M | EMB-NORMAL | propia, sin corrector | FastText **skipgram** sub 3-6 | 200 | A media | 0.6523 | 0.7818 | +0.0046 | nunca | CBOW, GloVe, epochs, lr, sample, corpus social |
| V5 bloque escalado | wikipedia_es 10M | EMB-NORMAL | propia | identica | 200 | A media, denso estandarizado | -- | 0.6901 | -0.0871 | nunca | estandarizar el denso: cerrado |
| V5 stacking | wikipedia_es 10M | EMB-NORMAL | propia | Ridge alpha=3 sobre 6 probabilidades OOF | 200 | A media | -- | 0.7678 | -0.0094 | nunca | stacking: cerrado, quinta vez que no aporta |
| V5 corpus web | mc4_es 10M | EMB-NORMAL + boilerplate C4 | propia | skipgram sub 3-6 | 200 | A media | 0.6638 | 0.7812 | +0.0040 | nunca | corpus de registro social |
| V5 corpus mixto | wiki 10M + mc4 10M = 20M | idem | propia | skipgram sub 3-6 | 200 | A media | 0.6658 | 0.7776 | +0.0004 | nunca | -- |
| V5 escala 50M | wikipedia_es 50M | EMB-NORMAL | propia | skipgram sub 3-6 | 200 | A media | 0.6687 | 0.7803 | +0.0031 | nunca | 100M: criterio de parada cumplido |
| V5 escala 5M | wikipedia_es 5M | EMB-NORMAL | propia | skipgram sub 3-6 | 200 | A media | 0.6670 | 0.7757 | -0.0015 | nunca | -- |
| V5 control word2vec | mc4_es 10M | EMB-NORMAL | propia | skipgram **minn=0 maxn=0** | 200 | A media | -- | 0.7762 | -0.0010 | nunca | word2vec solo se uso como control |
| V5.1 vector_size | wikipedia_es 10M | EMB-NORMAL | propia | skipgram sub 3-6 | 100 | A media | 0.6420 | 0.7823 | +0.0050 | nunca | 400d, 500d, 768d: cerrada por inercia |
| V5.1 vector_size | wikipedia_es 10M | EMB-NORMAL | propia | skipgram sub 3-6 | 300 | A media | 0.6516 | 0.7805 | +0.0033 | nunca | idem |
| V5.1 window | wikipedia_es 10M | EMB-NORMAL | propia | skipgram ws=**3** | 200 | A media | 0.6694 | **0.7830** | **+0.0057** | nunca | ws=1,2,9,15: cerrada |
| V5.1 window | wikipedia_es 10M | EMB-NORMAL | propia | skipgram ws=7 | 200 | A media | 0.6498 | 0.7806 | +0.0034 | nunca | idem |
| V5.1 min_count | wikipedia_es 10M | EMB-NORMAL | propia | skipgram minCount=2 | 200 | A media | 0.6516 | 0.7771 | -0.0001 | nunca | minCount=1, 8: cerrada |
| V5.1 min_count | wikipedia_es 10M | EMB-NORMAL | propia | skipgram minCount=5 | 200 | A media | 0.6547 | 0.7792 | +0.0020 | nunca | idem |
| V5.1 subwords | wikipedia_es 10M | EMB-NORMAL | propia | skipgram min 3 **max 5** | 200 | A media | 0.6397 | 0.7791 | +0.0018 | nunca | 2-6, 3-8, 4-6: cerrada |
| V5.1 subwords | wikipedia_es 10M | EMB-NORMAL | propia | skipgram min 3 **max 7** | 200 | A media | 0.6465 | 0.7767 | -0.0005 | nunca | idem |
| V5.1 negative | wikipedia_es 10M | EMB-NORMAL | propia | skipgram **neg=5** | 200 | A media | 0.6664 | 0.7811 | +0.0039 | nunca | neg=15, 20: cerrada |
| V5.1 comparacion final | -- | -- | -- | consolida 14 configuraciones | -- | -- | 0.6397-0.6694 | 0.7767-0.7830 | -- | **ninguna** | -- |

**14 configuraciones de embeddings, 0 adoptables, 0 submissions.** En la historia
entera del proyecto **ninguna** configuracion de embedding llego al leaderboard.

### 2.1 Arquitectura de referencia (una sola, en las 20 filas)

```
FastText C++ commit 1142dc4c4ecb (MIT), modo no supervisado
algoritmo    = skipgram (sg=1)             <-- UNICA variante entrenada
dim          = 200                         (100 y 300 probadas)
ws           = 5                           (3 y 7 probadas)
minCount     = 3                           (2 y 5 probadas)
minn/maxn    = 3/6                         (3/5 y 3/7 probadas)
neg          = 10                          (5 probada)
epoch        = 5                           <-- NUNCA VARIADO
alpha (lr)   = 0.05 (defecto de fastText)  <-- NUNCA VARIADO
sample       = 1e-4 (defecto)              <-- NUNCA VARIADO
bucket       = 200000
seed         = 42
hilos        = 16
throughput   = ~930k tok/s
```

### 2.2 Corpus (5 construidos, todos ya en disco)

| corpus | tokens | docs | tipos | cobertura TOXIC por tipo | por token |
|---|---:|---:|---:|---:|---:|
| wikipedia_es 5M | 5.011.404 | 1.694 | 173.942 | 0.6302 | 0.8646 |
| wikipedia_es 10M | 10.002.400 | 2.847 | 271.209 | 0.6839 | 0.8833 |
| mc4_es 10M | ~10M | -- | -- | **0.8256** | **0.9512** |
| mixto 20M | 20.000.045 | -- | -- | **0.8413** | **0.9553** |
| wikipedia_es 50M | 50.000.000 | 21.529 | 646.957 | 0.7703 | 0.9238 |

Auditoria de fuga ya realizada: hash de 8-gramas train/test contra el corpus, con
**0 textos de TOXIC** en el corpus.

---

## 3. Tres partes del encargo que el historial ya refuta

Esta es la conclusion mas importante de P0, y es la que condiciona P1 a P8.

### 3.1 La seccion 5: el corpus NO se uso "practicamente en bruto"

La premisa del encargo es **incorrecta como afirmacion de hecho**. V5 implemento y
documento una normalizacion completa, en dos etapas.

**`limpiar_marcado(texto)`**, sobre el marcado MediaWiki crudo: elimina comentarios,
referencias, plantillas (con 8 pasadas de sustitucion hasta punto fijo), tablas,
enlaces de archivo, marcas de edicion y entidades HTML, y aplica `html.unescape`;
descarta lineas vacias o que empiezan por `=`, quita las marcas de lista `*#:;`, y
aplica una whitelist de caracteres permitidos.

**`normalizar_emb_normal(texto)`**, version `v5.0-EMB-NORMAL`, verificada token a token
sobre los 2.885 textos de TOXIC:

```python
texto = texto.lower()                      # minusculas CONSERVANDO tildes
texto = RE_URL.sub(" url ", texto)         # URLs -> marcador
texto = RE_HASHTAG.sub(" hashtag ", texto) # hashtags -> marcador
texto = RE_MENCION.sub(" usuario ", texto) # menciones -> marcador
texto = RE_NUMERO.sub(" numero ", texto)   # numeros -> marcador
return RE_ESPACIOS.sub(" ", texto).strip()
```

Documentado explicitamente: **sin corrector ortografico y sin reconstruccion de
tildes**. Ademas: deduplicacion por sha1, minimo 20 tokens por documento y, en el
corpus web, limpieza de boilerplate de C4 (lineas repetidas, lineas cortas sin
digitos, documentos con mas del 30 % de URL o mas del 25 % de digitos).

Lo que el si se hizo y el encargo no propone: un diagnostico que **refuto** que la
brecha fuera ortografica por falta de tildes (el plegado a ASCII cubre solo el 8,5 %
de los OOV) y senalo que los tipos ausentes son evasiones que pierden vocales
(`callate`, `violacin`) y formas coloquiales o elongadas.

**Lo que si queda genuinamente sin hacer**, y es lo unico de la seccion 5 que
sobrevive: la comparacion **RAW / LIGHT / MEDIUM** como niveles de limpieza. Existe
una sola variante (EMB-NORMAL); no se midio nunca si *mas* limpieza o *menos* limpieza
ayudaria. La hipotesis de la seccion 5 no es "el corpus estaba sucio", sino "el nivel
de limpieza no se ha comparado". Eso es mas estrecho, y sigue siendo valido.

### 3.2 La seccion 11 (weighted mean frente a mean): ya probada y descartada

EBD-05 evaluo tres poolings sobre FastText 10M con LR, con rejilla completa de
umbrales (51,7 s):

| pooling | definicion | corrected_cv |
|---|---|---:|
| **A media** | media de los vectores de token | **0.660313** |
| B ponderado | media ponderada por TF-IDF, `(valores * pesos).sum(0) / pesos.sum()` | menor que A |
| C media + max | concatenacion de ambas | menor que A |

El veredicto registrado fue "A >= B/C": **el mean pooling gana al ponderado por
TF-IDF**. La pregunta de la seccion 11 tiene respuesta empirica desde V5, y la
respuesta es no. Reponerla seria repetir un experimento ya cerrado. Lo que si falta es
el ponderado por **IDF puro** normalizado, el **max solo**, y first/last, que son
variantes distintas de las tres medidas.

### 3.3 La seccion 13 (NB-SVM): ya implementado y con resultado lejos

`experiments/20_emocion_nbsvm_ortografia_v2.py` implemento E2-15 con log-count ratio
**fold-local** (smoothing `alpha_nb_svm=1.0`), en **seis** variantes:

| variante | corrected_cv | delta vs V03 |
|---|---:|---:|
| NB-UNION-LR (word+char, LR C=1.0) | 0.7612 | -0.0160 |
| NB-WORD-LR | 0.7538 | -0.0234 |
| NB-CHAR-LR | 0.7380 | -0.0392 |
| NB-UNION-SVC (LinearSVC C=1.0) | 0.7429 | -0.0343 |
| NB-WORD-SVC | menor | peor |
| NB-CHAR-SVC | menor | peor |

La implementacion es fiel a Wang & Manning: ratio fold-local, los dos predictores
lineales, y seis variantes que separan word de char y LR de SVC. Con una brecha de
-0.0160 frente a V03, **la linea NB-SVM no es competitiva** y no hay indicio de que
una reimplementacion mas cuidadosa cambie el orden de magnitud. Reabrirla seria gastar
presupuesto para un -0.016.

---

## 4. Lo que si es genuinamente nuevo

La lista que P4 a P7 pueden usar, y es mas corta de lo que el encargo supone.

### Arquitecturas nunca entrenadas

1. **CBOW.** Nunca. Las 20 filas son skipgram. Es la primera division real del encargo
   (seccion 8) y es nueva de verdad.
2. **GloVe entrenado desde cero.** Nunca. Tecnicamente distinto de fastText: cuenta
   coocurrencias globales y resuelve por factorizacion, no por optimizacion de
   ventana.
3. **Word2Vec como candidato, no como control.** Solo se uso con `minn=0 maxn=0` para
   aislar el efecto de las subpalabras, y perdio siempre (0.7762 frente a 0.7812). No
   se ha evaluado nunca con su propia rejilla como familia.

### Hiperparametros de fastText nunca variados

Ninguno de los tres aparece en las cinco familias de v5.1, que fueron `vector_size`,
`window`, `min_count`, `subwords` y `negative`.

4. **epoch.** Siempre 5. Con 5M-50M tokens y 200d, 5 epocas puede ser
   sistematicamente poco o demasiado: es el parametro de convergencia.
5. **alpha (learning rate).** Siempre el defecto 0.05.
6. **sample (subsampling de frecuencias).** Siempre el defecto 1e-4, es decir
   practicamente desactivado en un corpus donde las palabras funcionales dominan.

### Corpus nunca probado

7. **Registro social (tweets en espanol).** El diagnostico de v5 dice que el limite es
   el **registro**, no el tamano ni la fuente generica: web subio la cobertura por tipo
   de 0.6839 a 0.8256 y la aportacion **no** crecio (+0.0046 frente a +0.0040), y 50M
   de Wikipedia bajo la aportacion. La unica fuente que se ha aproximado al registro
   que la tarea pide es un corpus **de tweets en espanol**, y no se ha probado. Es la
   brecha mas grande que queda.

### Receptor y composicion nunca probados

8. **RidgeClassifier y SGDClassifier** como receptores del bloque denso. Solo LR y
   LinearSVC.
9. **SVD del bloque denso antes de unir** (200d reducidas a componentes, luego hstack).
   Nunca. Es distinto de unir el denso crudo, que es lo que se hizo y perdio.
10. **Pooling por IDF puro normalizado**, y **max solo**, first/last. A, B y C ya
    estan; B no es exactamente IDF.

---

## 5. El dato estructural que domina toda la linea

De `reports/embedding_architecture_v5_1.md`:

> El rango del embedding aislado es **4.8 veces mayor** que el de la fusion: la
> arquitectura cambia lo que se puede leer del embedding, pero casi nada de lo que
> aporta a la tarea.

- `emb_solo` se mueve entre 0.6397 y 0.6694 (rango **0.0297**).
- `fusion_cv` se mueve entre 0.7767 y 0.7830 (rango **0.0063**).

Y el motivo de cierre que registro v5 (EBD-13) es explicito:

> El limite es estructural: con 1.483 ejemplos de entrenamiento y 27.431 char
> n-grams, un vector denso de 200 numeros no tiene espacio para aportar.

Esto no se responde con un hiperparametro mejor. Se responde con un bloque denso que
sea mas pequeno y mas informativo, o con una fuente que cambie la geometria. Las
hipotesis 4 a 10 atacan lo segundo; solo la 9 ataca lo primero.

Una lectura honesta del patron: 14 cambios de arquitectura produjeron un `deltaOOF`
maximo de **+0.0057**, y el mejor `fusion_cv` de toda la fase fue **0.7830** frente al
**0.7772** de V03 en local. Son metricas distintas, pero el orden de magnitud es el que
es: el embedding mueve la metrica de la tarea en milesimas.

---

## 6. El cambio de criterio de adopcion, y lo que abre

El gate de v5 era de dos niveles, explicito y pre-registrado (D-039):

> delta >= +0.005 **Y** IC95 que excluya el 0

Con ese gate, las 14 configuraciones quedaron fuera. La seccion 14 de V21 cambia el
criterio:

> una mejora de leaderboard valida es suficiente para considerar adopcion. No exigir
> que el IC95 excluya cero como condicion absoluta.

**Consecuencia directa: el cambio de criterio reabre dos configuraciones que v5 cerro
solo por el IC95.**

| configuracion | delta | IC95 | por que la cerro v5 | bajo el criterio de V21 |
|---|---:|---|---|---|
| FUSION-WS-3 (window 3) | +0.0057 | [-0.002084, 0.013625] | IC95 toca el 0 | **reabierta** |
| FUSION-VS-100 (dim 100) | +0.0050 | [-0.00282, 0.013005] | IC95 toca el 0 | **reabierta** |

Ambas estan ya **entrenadas** (los `.vec` estan en disco con sha256 registrado), luego
reevaluarlas no cuesta tiempo de entrenamiento. Es la opcion de menor coste de toda
la fase.

### 6.1 Pero el historial competitivo de este proyecto no respalda gastar una submission asi

Este es el dato que hay que poner delante de cualquier decision de envio. Las cinco
sondas de V16 se enviaron con `deltaOOF` positivo y las cinco bajaron:

| submission | `local_f1` | LB | delta LB |
|---|---:|---:|---:|
| v16_ch1 tasa optima transportada | 0.778280 | 0.77828 | **-0.00478** |
| v16_ch5 tasa optima absoluta | -- | 0.77495 | **-0.00811** |
| v16_ch7 C=1.5 | 0.776653 | 0.77296 | **-0.01010** |
| v16_ch8 C=2.5 | 0.776752 | 0.77707 | **-0.00599** |
| v16_ch9 min_df=1 | 0.765532 | 0.76978 | **-0.01328** |
| V20 regla cross-score (ref 56597267) | 0.777998 | 0.77663 | **-0.00643** |

Y el balance global de `submissions/leaderboard_log.csv`, con 21 filas:

- **19 de 21 submissions quedaron por debajo de V03** (0.78306).
- Las **2 unicas que subieron** fueron v10 (+0.00047) y v11 (+0.00184), ambas de las
  fases tempranas, y **ninguna de las dos alcanzo a V03**: 0.77978 y 0.78150.
- En la ventana reciente, de V16 a V20, la cuenta es **7 de 7 a la baja**.

Escala de sensibilidad medida: unos **107 cambios netos de celda** equivalen a
**0.003** de LB; el error estandar del LB con 1.154 filas es **0.01219**. Un
`deltaOOF` de +0.0057 esta dentro del ruido del LB por un factor de ~2.

**Consecuencia operativa**: reabrir WS-3 o VS-100 es barato como *medicion* y caro
como *submission*. Solo justifica un envio si la diferencia de la candidata respecto
de V03 se apoya en un recuento de celdas favorable, no en un IC95 que toca el 0.

---

## 7. Conclusion de P0

1. Las reglas permiten la fase, con una ambiguedad abierta (lexicon externa) que se
   asume en el sentido mas restrictivo.
2. La premisa de la seccion 5 (corpus en bruto) es **incorrecta**: hubo
   `limpiar_marcado`, `EMB-NORMAL`, deduplicacion, filtro de longitud y auditoria de
   fuga. Lo unico no hecho es comparar niveles de limpieza (RAW/LIGHT/MEDIUM), que es
   una hipotesis mas estrecha y sigue viva.
3. La seccion 11 (weighted mean) y la seccion 13 (NB-SVM) estan **cerradas por
   evidencia**, no por falta de intento. No se reabren.
4. Lo genuinamente nuevo se reduce a **diez hipotesis** (seccion 4): tres de
   arquitectura (CBOW, GloVe, word2vec como candidato), tres hiperparametros de
   fastText nunca tocados (epoch, alpha, sample), una de fuente (corpus de registro
   social) y tres de receptor o composicion (Ridge/SGD, SVD del denso, pooling por
   IDF).
5. El cambio de criterio de la seccion 14 reabre **dos** configuraciones ya
   entrenadas, a coste cero de entrenamiento.
6. El limite documentado de la linea es **estructural** (dimension efectiva de los
   datos frente a densidad del bloque disperso), no de hiperparametros. Por eso los
   hiperparametros que faltan (epoch, alpha, sample) tienen baja probabilidad, y la
   que mas tiene es la fuente de registro social.

**P0 no ejecuta nada.** P1 (estado del arte) esta en curso. **P2 (auditoria de
limpieza) debe reformularse** a la comparacion de niveles RAW / LIGHT / MEDIUM sobre
el pipeline EMB-NORMAL ya existente, no a "aplicar limpieza" como si no se hubiera
hecho.
