# Especificación maestra — Fase v2 «Advanced Research»

Documento de referencia del proyecto `KAGGLE_TOXIC_2026_PUJ`.
Fuente: instrucción de Carlos Garnica para la fase v2 de investigación avanzada.
Estado: ejecutada y cerrada el 2026-09-25 mediante los paquetes P0–P8.
Trazabilidad: `reports/spec_compliance_v2.md`, `DECISIONS.md` (D-016…D-022),
`reports/advanced_research_v2.md` y `experiments.csv`.

---

# OBJETIVO: ENCONTRAR NUEVA SEÑAL Y SUPERAR EL LB 0.78306

============================================================
0. ROL
======

Actúa como un investigador senior de:

* NLP
* computational linguistics
* text mining
* stylometry
* hate speech detection
* abusive language detection
* multi-label classification
* statistical learning
* feature engineering
* information retrieval
* experimental design
* error analysis
* model calibration
* classical machine learning
* deep learning desde cero

No actúes como un optimizador ciego de hiperparámetros.

Tu objetivo es descubrir información nueva en el corpus que el sistema actual no esté explotando correctamente.

============================================================

1. COMPETENCIA
   ============================================================

Competencia:

https://www.kaggle.com/competitions/tweet-toxic-2026-puj/overview

Proyecto:

`KAGGLE_TOXIC_2026_PUJ`

Deadline:

2026-10-19

============================================================
2. BASELINE CONGELADA
=====================

La mejor submission conocida actualmente es:

`0.78306`

Correspondiente a:

TF-IDF word(1,2)
+
TF-IDF char_wb(3,5)
+
LogisticRegression balanceada
+
C=2.0
+
thresholds:

HS = 0.44
TR = 0.51
AG = 0.46

Submission de referencia:

`v03`

Esta versión debe quedar absolutamente congelada.

NO modificarla.

Crear:

`experiments/baseline_v03_frozen/`

Guardar:

* pipeline
* vectorizadores
* modelos
* thresholds
* seed
* CV
* OOF predictions
* submission
* hashes

Todos los experimentos nuevos deben compararse contra:

`V03_BASELINE`

============================================================
3. RESULTADOS ACTUALES QUE DEBES TOMAR COMO EVIDENCIA
=====================================================

Dataset:

Train = 1.731
Test = 1.154

Tres etiquetas:

HS
TR
AG

Prevalencias train:

HS = 43.67 %
TR = 29.98 %
AG = 38.24 %

Dependencias:

TR=1 solo ocurre cuando HS=1.

HS-TR phi ≈ 0.743

HS-AG phi ≈ 0.831

TR-AG phi ≈ 0.759

Esta estructura NO debe ignorarse.

Hasta ahora se probaron:

* TF-IDF word
* TF-IDF char
* word + char
* LogisticRegression
* LinearSVC
* ComplementNB
* ensembles
* Optuna
* features densas
* modelos ordinales/hierárquicos
* threshold tuning
* MLP
* CharCNN
* BiLSTM
* post-procesamiento jerárquico

Resultado ganador:

word + char + LogisticRegression + thresholds

============================================================
4. LIMITACIONES IMPORTANTES YA DEMOSTRADAS
==========================================

Ya se encontró:

* Char TF-IDF solo no gana.
* Word TF-IDF solo no gana.
* La unión word+char sí gana fuertemente.
* Features densas simples concatenadas degradan.
* Deep learning secuencial desde cero no supera al baseline.
* Ensemble indiscriminado no garantiza mejora.
* Optuna produjo mejor OOF pero peor leaderboard.
* Reducir capacidad no solucionó el gap.
* El post-proceso duro HS=0 => TR=0 y AG=0 degradó.
* Los near-duplicates existen.
* Los pseudo-autores contaminan parcialmente el split histórico.

NO repetir estas líneas sin una hipótesis nueva.

============================================================
5. CORRECCIÓN METODOLÓGICA OBLIGATORIA
======================================

El análisis anterior encontró:

Split basado únicamente en near-duplicates:

OOF ≈ 0.7927–0.7989 según configuración

Control con:

near-duplicates + pseudo-author

produjo aproximadamente:

OOF = 0.7838

Por tanto:

PROTOCOLO PRINCIPAL DE NUEVA INVESTIGACIÓN:

`StratifiedGroupKFold`

groups:

`near_duplicate_group + pseudo_author_group`

NO reutilizar el split histórico como única evidencia.

Sin embargo, conservar el resultado histórico para comparar.

Cada resultado nuevo debe reportar:

HISTORICAL_CV
CORRECTED_CV

La selección real debe basarse en:

CORRECTED_CV

============================================================
6. REGLA ABSOLUTA CONTRA LEAKAGE
================================

NO utilizar TEST para:

* fitting
* TF-IDF vocabulary
* feature selection
* target encoding
* lexicon induction
* threshold tuning
* model selection
* hyperparameter tuning
* calibration
* prototype construction
* class likelihood estimation

Todo lo aprendido debe ajustarse dentro del fold.

Especialmente:

* target encoding
* class lexical profiles
* pseudo-label statistics
* mention statistics
* hashtag statistics
* domain statistics
* entity statistics
* learned lexicons

TODO debe ser OOF o fold-local.

============================================================
7. REGLAS DE MODELOS
====================

NO:

* BERT
* RoBERTa
* DeBERTa
* DistilBERT
* ELECTRA
* GPT
* LLM
* transformers
* pretrained text classifiers
* pretrained word embeddings
* pretrained sentence embeddings
* GloVe
* pretrained Word2Vec
* pretrained FastText

Deep learning:

permitido solamente desde cero.

Lexicones:

pueden utilizarse si las reglas de la competencia lo permiten.

Deben documentarse.

spaCy u otra herramienta lingüística:

solo utilizar información simbólica como:

* POS
* lemas
* dependencias
* morfología

NO utilizar:

* `.vector`
* `.tensor`
* embeddings
* similarity basada en embeddings

============================================================
8. OBJETIVO DE LA NUEVA FASE
============================

La pregunta no es:

"¿qué modelo es mejor?"

La pregunta es:

"¿Qué señal lingüística o estructural está faltando?"

La secuencia debe ser:

OBSERVACIÓN

→ HIPÓTESIS

→ FEATURE / REPRESENTACIÓN

→ EXPERIMENTO

→ RESULTADO

→ ERROR ANALYSIS

→ ABLATION

→ DECISIÓN

============================================================
9. NO ASUMIR QUE EL PROBLEMA ES IRREDUCIBLE
===========================================

No utilizar como conclusión:

"el dataset tiene un techo de información"

sin evidencia.

La conclusión válida debe ser:

"Las representaciones y arquitecturas evaluadas hasta ahora no encontraron una señal adicional suficientemente robusta."

Esto deja abiertas nuevas representaciones.

============================================================
10. FASE 0 — RECUPERAR TODO EL CONOCIMIENTO EXISTENTE
=====================================================

Antes de experimentar:

leer:

`PROJECT_STATE.md`
`DECISIONS.md`
`RUNLOG.md`
`experiments.csv`
`FINAL_REPORT.md`

Además leer:

`reports/error_analysis.md`
`reports/eda_resumen.md`
`reports/features_densas_resultados.json`
`reports/estado_arte.md`
`reports/hipotesis.md`
`reports/control_pseudo_autores.json`

Inspeccionar:

* notebooks
* scripts
* OOF predictions
* submission logs
* feature catalogs
* resultados de Optuna
* resultados de deep learning

No asumir que algo fue probado porque aparece mencionado en el informe.

Verificarlo en los artefactos.

============================================================
11. FASE 1 — GAP ANALYSIS DEL FEATURE SPACE
===========================================

Crear:

`reports/advanced_gap_analysis_v2.md`

Construir una matriz:

phenomenon
detected_in_EDA
represented_in_model
representation
explicit_feature
supervised_feature
tested
result
remaining_gap
new_hypothesis

Investigar qué fenómenos del corpus están:

* solo descritos
* representados indirectamente
* representados parcialmente
* nunca modelados

Especial atención a:

* targeting
* insult
* profanity
* attribution
* quotation
* negation
* collective references
* person references
* social interaction
* emotional language
* sentence structure
* syntax
* morphology
* punctuation
* orthography
* lexical variation
* URL context
* hashtag context
* mention context

============================================================
12. FASE 2 — EL PROBLEMA NO ES SOLO MULTI-LABEL; EXISTEN DEPENDENCIAS
====================================================================

El sistema actual entrena:

HS
TR
AG

como clasificadores independientes.

Esto puede estar desperdiciando estructura.

Investigar:

### A. CLASSIFIER CHAINS

Orden:

HS → TR → AG

HS → AG → TR

AG → HS → TR

TR → HS → AG

Pero no limitarse a pasar una predicción binaria.

Utilizar:

`P(previous_label=1)`

como feature.

Ejemplo:

TR model:

TEXT FEATURES
+
P(HS=1 | text)

AG model:

TEXT FEATURES
+
P(HS=1 | text)
+
P(TR=1 | text)

Esto debe hacerse con predicciones OOF.

### B. SOFT LABEL CHAIN

No usar:

`HS predicted = 0`

como compuerta dura.

Usar:

`P(HS=1)`

como señal continua.

Esto evita el fallo observado en el post-proceso jerárquico.

### C. CONDITIONAL AG

Investigar:

P(AG | X, HS)

y:

P(AG | X, HS, TR)

comparado con:

P(AG | X)

### D. CONDITIONAL TR

Investigar:

P(TR | X, HS)

donde el modelo recibe evidencia probabilística de HS.

Registrar si esto mejora TR sin degradar HS.

============================================================
13. FASE 3 — MODELO DEL ESPACIO DE ETIQUETAS
============================================

Existen las combinaciones:

000
111
101
100
110
001
010 = inexistente

Investigar:

### Label powerset parcial

Modelar las combinaciones válidas.

Pero tener en cuenta:

* 28 casos de 110
* 26 casos de 001
* 92 casos de 100

No asumir que estas clases minoritarias soportarán un modelo directo.

Posibles soluciones:

* agrupamiento de clases raras
* jerarquía
* probabilidades estructuradas
* mixture of binary heads

La pregunta es:

¿podemos modelar la dependencia entre etiquetas sin perder la capacidad de cada cabeza?

============================================================
14. FASE 4 — EDA DE SEGUNDA GENERACIÓN
======================================

El EDA anterior fue bueno pero ahora debe orientarse exclusivamente hacia los errores.

Crear análisis por:

HS
TR
AG

y por combinaciones:

000
111
101
100
110
001

Analizar:

* longitud
* caracteres
* tokens
* vocabulario
* punctuation
* uppercase
* mentions
* hashtags
* URLs
* repeated characters
* laughter
* emoticons
* insults
* collective terms
* second person
* quotation
* negation
* imperatives
* names
* pronouns
* lexical rarity

No solo:

"está asociado".

Buscar:

"¿qué combinación de señales caracteriza cada régimen?"

============================================================
15. FASE 5 — INTERACCIONES LINGÜÍSTICAS
=======================================

No limitarse a contar:

`has_insult`

Investigar combinaciones.

Por ejemplo:

INSULT
×
SECOND_PERSON

INSULT
×
MENTION

INSULT
×
COLLECTIVE_REFERENCE

INSULT
×
LAUGHTER

INSULT
×
AFFECTION

INSULT
×
QUOTATION

INSULT
×
NEGATION

INSULT
×
RT

Esto es especialmente importante porque el error analysis ya encontró:

insulto + mención
puede ser:

odio

o:

broma/afiliación.

El modelo necesita aprender la diferencia.

============================================================
16. FASE 6 — BROMA / AFILIACIÓN
===============================

Este fenómeno ya apareció como principal fuente de falsos positivos.

No volver a utilizar exactamente las mismas features densas.

Investigar una representación estructural diferente.

Detectar:

* laughter
* affection
* possessives
* endearments
* positive polarity
* vocatives
* first/second person
* diminutives
* repetition
* emoticons

Crear patrones:

`insult + laughter`

`insult + affection`

`insult + positive cue`

`insult + vocative`

`insult + first person affection`

Ejemplo conceptual:

`"mi X"`
`"te amo"`
`"jajaja"`

No hardcodear ejemplos.

Crear detectores generales.

Y después:

`insult_affiliation_score`

mediante features aprendidas del corpus.

============================================================
17. FASE 7 — DISTANCIA ENTRE TARGET E INSULTO
=============================================

Esta línea debe investigarse explícitamente.

Para cada mención/persona/grupo y cada insulto detectado:

calcular:

* token distance
* character distance
* insult before target
* insult after target
* nearest target-insult pair
* average target-insult distance
* minimum target-insult distance

Crear:

`target_insult_distance`

`target_insult_order`

`target_insult_density`

Esto puede mejorar la distinción entre:

* insulto dirigido
* insulto general
* insulto citado
* insulto relacionado con un grupo

============================================================
18. FASE 8 — TARGET PROFILING
=============================

El análisis anterior mostró que TR depende fuertemente del targeting.

Expandir el concepto.

Detectar:

* persona
* grupo
* colectivo
* segunda persona
* primera persona
* organización
* gentilicio
* categoría social

Crear features:

* person-target score
* collective-target score
* target ambiguity
* target count
* target type
* number of possible targets
* target confidence

Además:

`individual_target × insult`

`collective_target × insult`

`second_person × insult`

============================================================
19. FASE 9 — CITATION / ATTRIBUTION
===================================

H13 estaba pendiente.

Debe ejecutarse.

Buscar patrones de:

* quotation marks
* reported speech
* colon + quote
* "dijo"
* "dice"
* "según"
* "comentó"
* "afirmó"
* "llamó"
* "acusó"
* "declaró"
* estructuras RT
* atribución a terceros

No asumir que una palabra de atribución implica cita.

Construir features:

* quotation_present
* quotation_count
* attributed_speech
* attribution_marker
* citation_score
* text_inside_quotes
* insult_inside_quotes
* insult_outside_quotes
* target_inside_quotes

Especialmente:

`insult_inside_quote`

porque puede diferenciar:

"X dijo 'insulto'"

de:

"insulto directo del autor"

y esto puede afectar HS/AG.

============================================================
20. FASE 10 — NEGACIÓN
======================

Investigar:

* "no"
* "nunca"
* "jamás"
* negaciones verbales
* negación + insulto
* negación + odio
* negación + targeting

Ejemplos conceptuales:

"no es un idiota"

vs

"es un idiota"

No utilizar reglas rígidas únicamente.

Construir:

* negation_count
* insult_under_negation
* negation_before_insult
* target_under_negation
* sentiment_under_negation

Cuando sea posible, utilizar dependencias sintácticas.

============================================================
21. FASE 11 — POS Y MORFOLOGÍA
==============================

El POS fue una hipótesis no ejecutada.

Debe evaluarse ahora.

Utilizar POS de manera simbólica.

Construir:

* POS frequencies
* POS ratios
* POS entropy
* POS bigrams
* POS trigrams
* POS 4-grams cuando tenga sentido
* pronoun categories
* verb/noun ratio
* adjective/noun ratio
* adverb ratio
* determiner ratio

Además:

morphology:

* number
* gender
* tense
* mood
* person
* degree

Evaluar si aportan información independiente del TF-IDF.

============================================================
22. FASE 12 — DEPENDENCY FEATURES
=================================

Explorar dependencias sintácticas.

Por ejemplo:

* subject → insult predicate
* pronoun → verb
* target noun → adjective
* target → nominal modifier
* target → apposition
* negation → insult
* vocative-like structures
* direct object → abusive predicate

Representarlas como tokens simbólicos:

`PRON_VERB`
`NOUN_ADJ`
`TARGET_ADJ`
`NEG_INSULT`

NO utilizar embeddings.

============================================================
23. FASE 13 — PATTERNS MORFOLÓGICOS
===================================

Construir:

* suffix patterns
* prefix patterns
* word endings
* word starts
* repeated morphological fragments
* adjective endings
* nominal endings
* inflection patterns

En español, investigar especialmente morfología que pueda distinguir:

* colectivos
* gentilicios
* adjetivos despectivos
* insultos derivados

No hardcodear listas sin justificación.

============================================================
24. FASE 14 — EVASIÓN ORTOGRÁFICA
=================================

El corpus tiene:

* elongations
* punctuation repetition
* altered profanity

Investigar normalizaciones paralelas.

IMPORTANTE:

NO reemplazar el texto original.

Crear distintos canales:

CHANNEL A:
raw text

CHANNEL B:
collapse repeated characters

CHANNEL C:
remove punctuation inserted inside words

CHANNEL D:
normalize common leetspeak

CHANNEL E:
whitespace normalization

CHANNEL F:
normalized profanity representation

La hipótesis:

una forma normalizada puede recuperar el lexema oculto mientras el raw channel conserva estilo.

Comparar:

raw only
normalized only
raw + normalized

============================================================
25. FASE 15 — PROFANITY / INSULT DETECTION
==========================================

El lexicón inducido actual tiene aproximadamente 150 lemas.

No asumir que eso es suficiente.

Construir dos enfoques:

### LÉXICO INDUCIDO

solo train/fold.

### LEXICÓN EXTERNO PERMITIDO

solo si las reglas de la competencia lo permiten.

Para cualquier lexicón externo documentar:

* fuente
* idioma
* licencia
* fecha
* categoría
* cantidad
* normalización
* leakage risk

Normalizarlo al alfabeto ASCII del corpus.

No utilizar un clasificador externo.

============================================================
26. FASE 16 — LEXICÓN DE EMOCIONES
==================================

H5/E04 nunca fue realmente probada.

Debe investigarse.

Solo usar:

* lexicones de emociones
* recursos léxicos permitidos

No usar modelos de sentimiento preentrenados.

Features:

* positive words
* negative words
* anger
* disgust
* fear
* sadness
* joy
* emotion density
* emotion entropy
* dominant emotion
* emotional polarity

Pero no añadir todo directamente.

Evaluar incrementalmente.

============================================================
27. FASE 17 — SENTIMIENTO LOCAL AL TARGET
=========================================

No limitarse a:

sentiment(tweet)

Investigar:

sentiment(target context)

Definir ventanas:

* ±3 tokens
* ±5 tokens
* ±10 tokens

respecto al target.

Features:

* local polarity
* local negative density
* local anger
* local disgust
* local insult density

Esto puede ser más útil para TR/AG que el sentimiento global.

============================================================
28. FASE 18 — TEXTOMETRÍA AVANZADA
==================================

Expandir las features densas sin repetir las ya probadas.

Para cada tweet investigar:

LONGITUD

* chars
* tokens
* words
* sentences

DISTRIBUCIONES

* mean
* median
* variance
* std
* coefficient of variation
* min
* max
* range
* Q1
* Q3
* IQR
* P10
* P90
* P95
* MAD
* skewness
* kurtosis

Estas estadísticas deben aplicarse a:

* word lengths
* token lengths
* sentence proxy lengths
* character runs
* punctuation runs
* token frequency
* repeated-token runs

No asumir que serán útiles.

============================================================
29. FASE 19 — ENTROPÍA Y CONCENTRACIÓN
======================================

Investigar:

* lexical entropy
* character entropy
* punctuation entropy
* POS entropy
* hashtag entropy
* mention entropy
* emotion entropy
* profanity entropy

Además:

* Simpson
* HHI
* concentration ratio
* Gini si procede

Pregunta:

¿los textos tóxicos muestran concentración de vocabulario, estilo o puntuación?

============================================================
30. FASE 20 — RAREZA LÉXICA
===========================

Construir frecuencias SOLO con fold train.

Features:

* mean log frequency
* median log frequency
* rare-token ratio
* very-rare ratio
* OOV
* unseen word ratio
* lexical familiarity
* lexical novelty

Después analizar:

rarity × toxicity

rarity × targeting

rarity × aggression

============================================================
31. FASE 21 — CLASS-CONDITIONAL LEXICAL PROFILES
================================================

Construir dentro de cada fold:

P(token | HS)
P(token | TR)
P(token | AG)

y:

log-odds

con smoothing.

Para cada tweet:

* HS lexical score
* TR lexical score
* AG lexical score
* margin
* entropy
* strongest class
* class disagreement

Además:

* HS vs non-HS
* TR vs non-TR
* AG vs non-AG

Y especialmente:

`HS lexical evidence`
+
`target evidence`

============================================================
32. FASE 22 — TARGET ENCODING DE ENTIDADES
==========================================

Investigar:

* mentions
* hashtags
* domains
* source accounts
* collective tokens

Construir:

P(label | entity)

con:

* smoothing
* minimum frequency
* unknown bucket

Características:

* mean score
* max score
* entropy
* margin
* count known
* unseen ratio

Todo OOF.

============================================================
33. FASE 23 — MENTION PROFILING
===============================

Las menciones son críticas.

Investigar:

* mention count
* unique mentions
* position
* first/last mention
* mention length
* repeated mention
* mention class profile

Pero especialmente:

`mention × insult`

`mention × second_person`

`mention × laughter`

`mention × affection`

`mention × collective_reference`

============================================================
34. FASE 24 — HASHTAG PROFILING
===============================

Analizar:

* hashtag vocabulary
* count
* density
* entropy
* rarity
* class profiles

Crear:

P(HS | hashtag)
P(TR | hashtag)
P(AG | hashtag)

con smoothing.

También estudiar:

* hashtag alone
* hashtag near insult
* hashtag near URL
* hashtag position

============================================================
35. FASE 25 — URL Y NOTICIAS
============================

El régimen noticia presentó error importante.

Separar:

* URL
* domain
* path
* t.co
* news-like patterns
* text before URL
* text after URL
* attribution
* quotation

Construir:

`news_regime_score`

NO asumir que una URL implica noticia.

Validarlo estadísticamente.

============================================================
36. FASE 26 — RETWEETS
======================

Separar:

PURE_RT

de:

RT_WITH_COMMENT

Features:

* RT prefix
* source mention
* source profile
* user-added text
* comment length
* insult in added text
* quotation
* attribution

Investigar específicamente:

¿el odio está en el contenido citado o en el comentario del usuario?

Esto puede ser crítico para HS/AG.

============================================================
37. FASE 27 — CONTENIDO CITADO VS CONTENIDO DEL AUTOR
=====================================================

Para:

RT
quotation
"según"
"dijo"
etc.

intentar dividir:

AUTHOR_TEXT

vs

QUOTED_TEXT

Crear:

* insult_in_author_text
* insult_in_quote
* target_in_author_text
* target_in_quote
* sentiment_author
* sentiment_quote

La hipótesis:

el mismo insulto tiene distinta interpretación dependiendo de quién lo está pronunciando.

============================================================
38. FASE 28 — WORD + CHAR + STRUCTURE
=====================================

El experimento anterior mostró:

word + char

es excelente.

Ahora NO asumir que las features estructurales deben concatenarse directamente.

Evaluar:

A. word + char

B. word + char + POS sparse

C. word + char + syntax sparse

D. word + char + lexical patterns

E. word + char + structural model scores

F. late fusion

No mezclar todo a la vez.

============================================================
39. FASE 29 — POS COMO REPRESENTACIÓN SPARSE
===========================================

Probar explícitamente:

POS sequence → CountVectorizer/TF-IDF

por ejemplo:

`PRON VERB ADJ`

y:

POS bigrams/trigrams.

Comparar:

POS only
word+POS
word+char+POS

La razón es capturar estructura gramatical de forma sparse en lugar de 15 números densos.

============================================================
40. FASE 30 — DEPENDENCY PATHS COMO FEATURES SPARSE
===================================================

Convertir patrones sintácticos a strings y utilizar:

CountVectorizer
TF-IDF
LinearSVC/LR

Ejemplos conceptuales:

`PRON→VERB`
`NOUN→ADJ`
`NEG→VERB`
`TARGET→ADJ`

No utilizar embeddings.

============================================================
41. FASE 31 — N-GRAMAS ALTERNATIVOS
====================================

El baseline usa:

word 1-2
char_wb 3-5

Explorar de forma controlada:

CHAR:

2-4
3-4
3-5
3-6
4-6
4-7
5-7

WORD:

1
1-2
1-3
2-3

y:

character `char`
character `char_wb`

No ejecutar todas indiscriminadamente.

Primero estudiar vocabulario y sparsity.

============================================================
42. FASE 32 — REPRESENTACIONES DE PESO ALTERNATIVAS
===================================================

Investigar:

* Count
* binary Count
* TF
* TF-IDF
* sublinear TF
* delta TF
* log-count ratio
* BM25-style weighting
* NB-SVM

Especialmente:

NB-SVM

para:

* word
* char
* char_wb

Comparar contra:

TF-IDF + LR.

============================================================
43. FASE 33 — NB-SVM
====================

Implementar correctamente:

log-count ratio

sobre train fold.

Después:

LinearSVC
LogisticRegression

Evaluar:

* standalone F1
* error correlation
* disagreement
* contribution to ensemble

Una mejora pequeña puede ser útil si aporta errores diferentes.

============================================================
44. FASE 34 — NORMALIZACIÓN MULTICANAL
======================================

No elegir una sola normalización.

Construir:

RAW

NORMALIZED

NO_PUNCT_INTERNAL

COLLAPSED_REPETITIONS

LEXICON_MASKED

y evaluar:

raw

raw + normalized

raw + normalized + pattern channel

La hipótesis:

la normalización puede recuperar lenguaje tóxico ofuscado sin perder la señal original.

============================================================
45. FASE 35 — FEATURE ENGINEERING DE PATRONES
=============================================

Crear patrones estructurados:

* target + insult
* target + adjective
* target + negative adjective
* second person + insult
* mention + insult
* collective + insult
* negation + insult
* quotation + insult
* laughter + insult
* affection + insult
* RT + insult
* URL + attribution

Estas features deben ser:

* binary
* count
* density
* distance
* position

No solamente booleanas.

============================================================
46. FASE 36 — MODELO POR ETIQUETA
=================================

Analizar qué señales necesita cada etiqueta.

HS:

focus:

* hate lexicon
* collective references
* derogation
* target
* implicit hate
* attribution

TR:

focus:

* person/group targeting
* mention
* second person
* target-insult distance
* directed constructions

AG:

focus:

* insult
* profanity
* intensity
* repetition
* anger/disgust
* aggressive syntax

No asumir que el mismo feature space óptimo para HS es óptimo para TR y AG.

Investigar:

representation_per_label

============================================================
47. FASE 37 — FEATURES ESPECÍFICAS POR ETIQUETA
===============================================

Comparar:

SHARED_FEATURES

vs:

LABEL_SPECIFIC_FEATURES

Ejemplo:

HS model:
word + char + collective + hate lexicon

TR model:
word + char + target features

AG model:
word + char + insult/emotion features

Esto puede ser más adecuado que compartir exactamente la misma representación.

============================================================
48. FASE 38 — THRESHOLD TUNING DE SEGUNDA GENERACIÓN
====================================================

Los thresholds:

HS 0.44
TR 0.51
AG 0.46

ya funcionan.

NO reoptimizar ciegamente.

Investigar:

* nested threshold tuning
* probability calibration
* class-specific thresholds
* threshold conditioned on uncertainty
* threshold conditioned on tweet type

Nunca calibrar con test.

============================================================
49. FASE 39 — THRESHOLDS POR REBANADA
====================================

H15 quedó pendiente.

Investigar thresholds diferentes para:

* hashtag-heavy
* mention-heavy
* URL-heavy
* RT
* short
* long
* quotation
* collective-reference
* question

Pero NO crear un threshold por cualquier slice pequeña.

Primero exigir:

n suficiente.

Utilizar shrinkage hacia el threshold global.

Ejemplo conceptual:

`threshold_slice = weighted(global, local)`

No utilizar threshold extremo para una slice con 20 ejemplos.

============================================================
50. FASE 40 — CALIBRACIÓN
=========================

Evaluar:

* Platt scaling
* isotonic regression cuando sea apropiado
* temperature scaling si se justifica

Medir:

* Brier
* calibration error
* F1
* threshold stability

La calibración es importante porque los decision scores entrarán en cadenas y meta-modelos.

============================================================
51. FASE 41 — SOFT LABEL DEPENDENCY
===================================

Construir un meta-modelo sobre:

HS score
TR score
AG score

más:

* HS margin
* TR margin
* AG margin
* HS entropy
* TR entropy
* AG entropy
* label disagreement

y features estructurales.

No usar las etiquetas reales de las capas anteriores.

Utilizar únicamente predicciones OOF.

============================================================
52. FASE 42 — META-MODEL
========================

Comparar:

* LogisticRegression
* Ridge
* shallow MLP

Input:

base model scores
+
conditional label scores
+
structural features
+
uncertainty

Objetivo:

predecir cada etiqueta.

NO utilizar un meta-modelo grande.

Con 1.731 ejemplos, regularizar fuertemente.

============================================================
53. FASE 43 — ERROR DE LAS TRES ETIQUETAS
=========================================

Ya se encontró aproximadamente:

101 ejemplos
≈ 5.8 %
fallan las tres etiquetas.

Estos ejemplos son prioritarios.

Crear:

`reports/all_three_errors.csv`

Analizar:

* common lexical profile
* target presence
* quotation
* attribution
* sentiment
* emotion
* length
* URL
* RT
* collective reference
* insult
* implicit hate

Pregunta:

¿qué fenómeno común explica que todos los modelos fallan?

============================================================
54. FASE 44 — ERRORES DE UNA SOLA ETIQUETA
==========================================

Analizar separadamente:

HS correct, TR wrong, AG correct

HS wrong, TR correct, AG correct

etc.

Esto revela qué información necesita cada cabeza.

============================================================
55. FASE 45 — FALSE POSITIVES
=============================

Separar:

FP-HS
FP-TR
FP-AG

y buscar:

* joke
* affiliation
* quotation
* non-hateful insult
* metaphor
* news
* third-party speech
* lexical ambiguity

Crear una taxonomía.

============================================================
56. FASE 46 — FALSE NEGATIVES
=============================

Separar:

FN-HS
FN-TR
FN-AG

buscar:

* implicit hatred
* collective target
* coded language
* no explicit profanity
* paraphrase
* attribution
* news reporting
* unusual syntax

============================================================
57. FASE 47 — ERROR BY LABEL COMBINATION
========================================

Para cada combinación:

000
111
101
100
110
001

calcular:

* precision
* recall
* F1
* text length
* lexical features
* target signals
* insult signals

La pregunta es:

¿Qué fenómeno diferencia:

111
vs
101

y:

100
vs
000

y:

110
vs
100

============================================================
58. FASE 48 — MODEL SPECIALIZATION SOLO SI HAY EVIDENCIA
========================================================

No crear especialistas automáticamente.

Primero demostrar:

`error_slice significantly differs from global`

Ejemplos:

short
RT
quotation
collective

Solo entonces entrenar specialist.

============================================================
59. FASE 49 — LATE FUSION
=========================

Dado que concatenar dense features degradó:

investigar late fusion:

MODEL_TEXT
MODEL_LEXICON
MODEL_SYNTAX
MODEL_TARGET
MODEL_EMOTION
MODEL_LABEL_DEPENDENCY

→ decision scores

→ meta-model

No concatenar todas las features originales.

============================================================
60. FASE 50 — DIVERSIDAD DE MODELOS
===================================

Para cada modelo:

calcular:

* disagreement
* pairwise error correlation
* boundary disagreement
* correction rate

Un modelo puede conservarse aunque tenga peor F1 si corrige errores que el baseline no corrige.

============================================================
61. FASE 51 — CONTRASTAR "MODELO PEOR PERO COMPLEMENTARIO"
==========================================================

Para cada candidato:

standalone F1

vs

ensemble gain.

Registrar:

`complementarity_gain`

No eliminar automáticamente un modelo con F1 menor.

============================================================
62. FASE 52 — BOOTSTRAP DE DIFERENCIAS
======================================

Como el corpus tiene solo 1.731 tweets:

para comparaciones importantes, utilizar bootstrap a nivel de tweet.

Calcular:

* delta F1
* intervalo de confianza
* distribución de diferencias

No utilizar bootstrap para justificar cualquier mejora.

Utilizarlo como evidencia adicional.

============================================================
63. FASE 53 — ABLATION
======================

Construir:

BASELINE V03

*

TARGET FEATURES

*

POS

*

SYNTAX

*

NEGATION

*

QUOTE/ATTRIBUTION

*

EMOTION

*

LEXICON

*

ORTHOGRAPHY

*

ENTITY PROFILING

*

DISTRIBUTIONAL FEATURES

*

LABEL DEPENDENCY

*

LATE FUSION

medidos de manera incremental.

============================================================
64. FASE 54 — FEATURE REDUNDANCY
================================

Para las features densas:

calcular:

* Pearson
* Spearman
* mutual information
* variance
* missingness
* redundancy

No asumir que una feature útil estadísticamente será útil predictivamente.

============================================================
65. FASE 55 — DATASET FINAL DE FEATURES
=======================================

Guardar:

`features/v2_structured_features.parquet`

y:

`reports/feature_catalog_v2.csv`

Cada feature debe tener:

name
family
formula
dtype
source
hypothesis
leakage_risk
fold_dependency
experiment_id

============================================================
66. FASE 56 — VERSIONES
=======================

No sobrescribir:

V03

Crear:

V04_LABEL_DEPENDENCY

V05_TARGETING

V06_SYNTAX

V07_LEXICON_EMOTION

V08_ATTRIBUTION

V09_DISTRIBUTIONAL

V10_NORMALIZATION

V11_NBSVM

V12_LATE_FUSION

V13_FINAL

Los nombres reales pueden adaptarse.

============================================================
67. FASE 57 — RESEARCH LOG
==========================

Actualizar después de cada experimento:

`RUNLOG.md`

y:

`experiments.csv`

Campos:

experiment_id
hypothesis
feature_family
representation
model
parameters
historical_cv
corrected_cv
fold_scores
bootstrap_delta
ensemble_gain
runtime
decision
status

============================================================
68. FASE 58 — DECISIONS
=======================

Cada decisión:

D-XXX

Observation:
...

Hypothesis:
...

Experiment:
...

Result:
...

Interpretation:
...

Decision:
...

Evidence:
...

============================================================
69. FASE 59 — RESEARCH BACKLOG
==============================

Crear:

`reports/research_backlog_v2.md`

Cada línea:

ID
hypothesis
signal
implementation
validation
expected_information
status
result

No usar "prioridad" sin justificación.

============================================================
70. FASE 60 — CONTROL COMPUTACIONAL
===================================

El dataset es pequeño pero las representaciones sparse pueden crecer.

No construir matrices enormes innecesariamente.

Reutilizar:

* tokenization
* frequency tables
* TF-IDF
* OOF predictions
* lexical profiles
* feature caches

Guardar artefactos.

============================================================
71. FASE 61 — TESTS UNITARIOS
=============================

Probar features con:

* texto corto
* texto vacío
* URL-only
* mention-only
* hashtag-only
* RT
* quotation
* insult
* negation
* repeated chars
* punctuation

Ninguna feature puede producir:

* NaN inesperado
* infinito
* leakage
* comportamiento dependiente del orden del dataset

============================================================
72. FASE 62 — DATA AUGMENTATION
===============================

Por defecto NO utilizar augmentation.

No usar:

* LLM paraphrasing
* backtranslation
* generación externa
* pseudo-labeling
* corpus adicional para entrenamiento

Solo investigar augmentation si las reglas lo permiten explícitamente y existe evidencia de que la transformación conserva la etiqueta.

Por defecto:

NO.

============================================================
73. FASE 63 — EXTERNAL LEXICONS
===============================

Puede evaluarse el uso de lexicones permitidos.

Pero:

* no utilizar modelos de clasificación externos
* no utilizar embeddings
* no introducir labels externas
* no utilizar corpus externos como entrenamiento

Para cada lexicon:

documentar:

source
language
license
categories
normalization
coverage
risks

Crear una variante autocontenida sin lexicon externo.

============================================================
74. FASE 64 — NORMALIZACIÓN DEL LEXICÓN
=======================================

Dado que el corpus es ASCII:

normalizar los lexicones para:

* case
* accents
* ñ
* punctuation

sin modificar el corpus de entrenamiento.

Comparar cobertura.

============================================================
75. FASE 65 — REPRESENTACIÓN DEL INSULTO
========================================

Investigar tres canales:

RAW INSULT

NORMALIZED INSULT

ABSTRACT INSULT

donde ABSTRACT puede representar:

`<INSULT>`

sin eliminar el raw text.

Hipótesis:

el canal abstracto puede aprender la estructura:

TARGET + INSULT + EMOTION

sin depender exclusivamente del token específico.

Construirlo con un léxico permitido y fold-local cuando dependa del target.

============================================================
76. FASE 66 — N-GRAMAS SOBRE PATRONES ESTRUCTURALES
===================================================

Convertir secuencias en tokens simbólicos:

`MENTION INSULT`

`SECOND_PERSON INSULT`

`NEGATION INSULT`

`TARGET COLLECTIVE`

`QUOTE INSULT`

`RT INSULT`

y aplicar:

TF-IDF

Esto puede representar composicionalidad sin transformers.

============================================================
77. FASE 67 — DISTANCIA / ORDEN / DENSIDAD
==========================================

Para cualquier patrón lingüístico importante calcular:

* count
* density
* normalized position
* first occurrence
* last occurrence
* distance
* order

No limitarse a binary indicators.

============================================================
78. FASE 68 — MODELO ESPECÍFICO PARA EL RÉGIMEN NOTICIA
=======================================================

El análisis anterior encontró un régimen:

URL/hashtag
≈ news-like

con F1 notablemente menor.

Investigar un modelo especializado para:

* quotation
* attribution
* named entity mention
* URL
* headline-like syntax
* third-person narration

Pero solo si:

`P(error | news_regime)`

es suficientemente diferente.

============================================================
79. FASE 69 — AGRESIVIDAD CONTRA GRUPOS
=======================================

Uno de los hallazgos más fuertes:

aproximadamente 45.5 % de los FN de AG tiene referencia colectiva

frente a aproximadamente 13.9 % de TP.

Esto debe investigarse como una línea independiente.

Construir:

* collective_present
* collective_count
* collective_strength
* collective + insult
* collective + negative adjective
* collective + second person
* collective + target
* collective lexical score
* collective toxicity score

Especialmente:

`collective + aggression`

============================================================
80. FASE 70 — HS IMPLÍCITO
==========================

Investigar casos:

HS=1
sin insulto explícito.

Clasificar los FN de HS:

* stereotype
* dehumanization
* fear
* exclusion
* negative generalization
* threat
* insinuation

Crear features para cada fenómeno si hay evidencia suficiente.

NO usar un detector preentrenado externo.

Puede usar:

* lexicon
* syntax
* handcrafted patterns
* train-induced vocabulary

============================================================
81. FASE 71 — DETECCIÓN DE ESTEREOTIPOS
=======================================

Buscar estructuras:

`group + negative predicate`

`group + adjective`

`group + generalized statement`

`all/every + group`

`group + stereotype`

La finalidad es capturar odio sin profanity.

============================================================
82. FASE 72 — PATRONES DE GENERALIZACIÓN
========================================

Features:

* universal quantifier
* plural target
* generic statement
* collective noun
* generalized adjective

Esto es especialmente relevante para HS.

============================================================
83. FASE 73 — TARGETED VS GENERAL
=================================

Construir un continuo:

general statement
→ collective target
→ specific target
→ direct address

en vez de:

TR yes/no

Usar scores:

`target_specificity`

`target_directness`

`target_collectivity`

como features auxiliares.

============================================================
84. FASE 74 — AGRESIVIDAD COMO CONTINUO
=======================================

Construir un vector:

* insult intensity
* profanity count
* punctuation intensity
* uppercase intensity
* repetition
* imperative
* anger
* disgust
* threat
* second-person

y generar:

`aggression_score`

No asumir que este score sustituirá AG.

Utilizarlo como feature.

============================================================
85. FASE 75 — HS SCORE
======================

Análogamente construir:

`hate_evidence_score`

basado en:

* target group
* hateful lexicon
* stereotype
* dehumanization
* exclusion
* generalization

y:

`target_evidence_score`

para TR.

============================================================
86. FASE 76 — TRES SCORES LATENTES
==================================

Crear conceptualmente:

HATE_EVIDENCE

TARGET_EVIDENCE

AGGRESSION_EVIDENCE

y analizar:

si estas dimensiones son independientes.

Luego usarlas como features para las tres cabezas.

============================================================
87. FASE 77 — INTERPRETABILIDAD
=================================

Para los modelos finales investigar:

* top positive features
* top negative features
* coefficient magnitude
* permutation importance
* feature group ablation

No interpretar coeficientes como causalidad.

============================================================
88. FASE 78 — MODELO FINAL POR ETIQUETA
=======================================

Considerar arquitectura:

HS:
text + hate evidence

TR:
text + target evidence + P(HS)

AG:
text + aggression evidence + P(HS) + P(TR)

Esto debe compararse contra:

HS
TR
AG

independientes.

============================================================
89. FASE 79 — META-ENSEMBLE
===========================

Si se construyen varios canales:

TEXT_MODEL

SYNTAX_MODEL

LEXICON_MODEL

TARGET_MODEL

EMOTION_MODEL

LABEL_DEPENDENCY_MODEL

usar:

OOF predictions

como features.

No concatenar todos los raw matrices.

============================================================
90. FASE 80 — STACKING CORRECTO
================================

El stacking debe cumplir:

Base models train fold
→ produce OOF predictions

Meta-model:
→ entrenado únicamente sobre OOF predictions

Para evaluación de un fold:

base models no deben haber visto ese fold.

No utilizar predicciones in-sample.

============================================================
91. FASE 81 — BOOTSTRAP DEL ENSEMBLE
====================================

Comparar:

baseline V03
vs
candidate ensemble

a nivel de tweet mediante bootstrap.

Medir:

mean delta
95% CI

No aceptar automáticamente una mejora cuyo CI sea completamente compatible con 0.

============================================================
92. FASE 82 — CRITERIO DE ADOPCIÓN
==================================

Para nuevas features/modelos:

ADOPTAR automáticamente si:

* mejora corrected CV
* mejora al menos 3 de 5 folds
* no empeora claramente la estabilidad
* no hay leakage

Para mejoras pequeñas:

si standalone:

ΔF1 < 0.005

pero:

ensemble_gain > 0.0015

y la complementariedad es consistente:

puede conservarse como componente complementario.

No gastar leaderboard para cualquier mejora local.

============================================================
93. FASE 83 — RESEARCH HOLDOUT
==============================

Si aún no existe un holdout independiente:

crear uno agrupado por:

near-duplicate + pseudo-author

Idealmente:

10–15 % de los grupos.

CONGELAR.

No usar para el desarrollo diario.

Solo para candidatos finales.

============================================================
94. FASE 84 — LEADERBOARD
=========================

El mejor LB actual:

0.78306

NO utilizarlo como validation set.

Una submission solo se realiza cuando:

1. existe hipótesis
2. corrected CV mejora
3. existe evidencia de complementariedad
4. candidato es diferente del baseline

No enviar cada variante.

============================================================
95. FASE 85 — ANALIZAR EL GAP LOCAL ↔ LEADERBOARD
================================================

Construir tabla:

model
corrected_cv
historical_cv
leaderboard
gap_corrected
gap_historical

Buscar:

¿qué características de modelos producen mejor transferencia?

No asumir que mayor CV significa mayor LB.

============================================================
96. FASE 86 — TEST PREVALENCE
==============================

No usar labels del test.

No inferir directamente sus etiquetas.

No optimizar thresholds contra la distribución del test.

No pseudo-labeling.

La única información permitida del test para la generación de submission es:

* aplicar `transform`
* ejecutar `predict`
* generar CSV

============================================================
97. FASE 87 — AUDITORÍA FINAL
=============================

Antes de considerar adoptado cualquier candidato:

LEAKAGE AUDIT

* no test fit
* no external labels
* no external corpus training
* no pretrained embeddings
* no pretrained classifiers
* no OOF contamination
* no target encoding leakage
* no meta-model leakage

RULE AUDIT

* transformers = 0
* LLM = 0
* pretrained embeddings = 0
* external classifier = 0

REPRODUCIBILITY AUDIT

* seeds
* requirements
* paths
* hashes
* artifact loading
* deterministic output

============================================================
98. FASE 88 — ESTRUCTURA DE ARTEFACTOS
======================================

Crear:

`reports/advanced_research_v2.md`

`reports/advanced_gap_analysis_v2.md`

`reports/research_backlog_v2.md`

`reports/error_intersection_v2.csv`

`reports/error_by_label_combination_v2.csv`

`reports/model_complementarity_v2.csv`

`reports/feature_catalog_v2.csv`

`reports/target_encoding_audit_v2.md`

`reports/rule_audit_v2.md`

============================================================
99. FASE 89 — FEATURE DATASETS
==============================

Guardar versiones:

`features/v2_textometry.parquet`

`features/v2_targeting.parquet`

`features/v2_syntax.parquet`

`features/v2_lexicon.parquet`

`features/v2_emotion.parquet`

`features/v2_entities.parquet`

`features/v2_label_dependency.parquet`

`features/v2_final_candidate.parquet`

No sobrescribir versiones previas.

============================================================
100. FASE 90 — EXPERIMENT LOG
=============================

Cada experimento:

ID
hypothesis
feature_family
representation
model
parameters
historical_CV
corrected_CV
fold_scores
bootstrap_delta
ensemble_gain
holdout
kaggle
runtime
decision

============================================================
101. FASE 91 — PRIORIDAD DE INVESTIGACIÓN
=========================================

NO significa "garantía de mejora".

Orden inicial:

1. reproducir V03
2. corrected CV
3. classifier chains
4. soft label dependencies
5. target-insult distance
6. quote/attribution
7. collective + aggression
8. POS sparse
9. dependency patterns
10. emotion lexicon
11. permitted external lexicons
12. lexical class profiles
13. mention/hashtag/domain profiling
14. orthographic multi-channel normalization
15. NB-SVM
16. structural pattern n-grams
17. label-specific feature spaces
18. slice-specific thresholds
19. calibration
20. late fusion
21. second-generation stacking
22. final candidate
23. frozen holdout
24. only then leaderboard submission

Después de obtener resultados, reajustar el orden según la evidencia.

============================================================
102. FASE 92 — NO REPETIR EXPERIMENTOS NEGATIVOS
================================================

Antes de ejecutar algo:

buscar en:

`experiments.csv`

y:

`DECISIONS.md`

Si ya fue probado:

determinar si la nueva propuesta es realmente distinta.

Ejemplo:

"features densas + TF-IDF"

ya fue negativo.

No repetir la misma concatenación.

En cambio:

"features sintácticas como sparse n-grams"

sí es una hipótesis nueva.

============================================================
103. FASE 93 — NO CONFUNDIR SIGNIFICANCIA CON UTILIDAD
======================================================

Una feature puede tener:

p < 0.001

y ser inútil predictivamente.

Evaluar siempre:

* effect size
* mutual information
* incremental F1
* complementarity

============================================================
104. FASE 94 — NO CONFUNDIR CORRELACIÓN CON CAUSALIDAD
=========================================================

Nunca escribir:

"la presencia de X causa toxicidad".

Utilizar:

"X está asociado con..."
"X proporciona señal discriminativa..."

============================================================
105. FASE 95 — RESPUESTA A RESULTADOS NEGATIVOS
===============================================

Cuando algo falle:

NO insistir ciegamente.

Preguntar:

¿falló la hipótesis?

¿falló la representación?

¿falló la integración?

¿hay una variante más adecuada?

Ejemplo:

Las features densas fallaron.

No concluir:

"las features lingüísticas no sirven."

Puede significar:

"estas features no funcionan bien concatenadas a la representación sparse."

============================================================
106. FASE 96 — CONCLUSIÓN DE CADA EXPERIMENTO
============================================

Después de cada experimento escribir:

OBSERVACIÓN

HIPÓTESIS

MÉTODO

RESULTADO

INTERPRETACIÓN

DECISIÓN

PRÓXIMO PASO

============================================================
107. FASE 97 — CRITERIO DE CIERRE
=================================

Cerrar una línea solo cuando:

* la hipótesis fue suficientemente probada
* las variantes razonables fueron evaluadas
* no existe mejora
* el resultado es estable

Ejemplo:

"POS sparse evaluado en E-071–E-075; no aporta mejora incremental ni complementariedad; línea cerrada."

============================================================
108. FASE 98 — NO DESTRUIR EL BASELINE
======================================

Si un candidato empeora:

volver a V03.

El sistema V03 siempre debe permanecer ejecutable.

============================================================
109. FASE 99 — CONDICIÓN DE FINALIZACIÓN
========================================

La fase no termina simplemente porque:

"probamos muchas cosas".

Termina cuando:

A. se encuentra una mejora robusta

o

B. las nuevas fuentes de señal razonables han sido evaluadas y documentadas

La meta principal es:

superar 0.78306

pero la evidencia experimental tiene prioridad sobre el número aislado.

============================================================
110. PRIMERA ACCIÓN
===================

Comienza inmediatamente.

NO empieces entrenando modelos nuevos.

Primero:

1. leer todos los artefactos existentes
2. congelar V03
3. reproducir V03
4. reconstruir corrected CV con:
   near-duplicate + pseudo-author
5. calcular baseline por etiqueta
6. construir error intersection
7. construir error-by-label-combination
8. auditar qué hipótesis realmente quedaron sin ejecutar
9. crear advanced_gap_analysis_v2.md
10. crear research_backlog_v2.md

Después iniciar:

PRIMERA LÍNEA:

soft label dependency / classifier chains

SEGUNDA:

target-insult distance + targeting structure

TERCERA:

quotation / attribution / negation

CUARTA:

POS/dependency sparse representations

QUINTA:

emotion / permitted lexicons

SEXTA:

orthographic multi-channel normalization

SÉPTIMA:

NB-SVM

OCTAVA:

late fusion / second-generation stacking

No saltar directamente a optimización masiva.

============================================================
111. REGLA ABSOLUTA
===================

No quiero:

"otro modelo"

Quiero:

"otra fuente de información".

No quiero:

"añadir 50 features"

Quiero:

"representar mejor un fenómeno lingüístico demostrado en el corpus".

No quiero:

"el leaderboard está en plateau"

Quiero:

"el sistema actual está limitado por estas señales; estas nuevas hipótesis atacan específicamente esas limitaciones".

No quiero:

"las features lingüísticas no funcionan"

Quiero:

"esta representación de features lingüísticas no añadió información bajo esta integración; se investiga otra representación".

No quiero:

"el error es irreducible"

Quiero:

"los experimentos realizados hasta ahora no encontraron señal adicional robusta".

============================================================
112. OBJETIVO FINAL DEL PROYECTO
================================

Construir una versión:

`TOXIC_ADVANCED_V01`

que pueda demostrar:

* qué señal adicional fue descubierta
* por qué esa señal es pertinente
* cómo se representa
* cómo evita leakage
* qué mejora produce
* en qué clases ayuda
* qué errores corrige
* qué errores nuevos introduce
* por qué el modelo final es mejor que V03

El resultado final debe ser defendible académicamente y reproducible de extremo a extremo.
