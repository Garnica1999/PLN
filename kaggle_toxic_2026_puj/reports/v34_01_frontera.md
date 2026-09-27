# V34-01: auditoria del espacio experimental de TOXIC

Generado por `experiments/v34_01_frontera.py` (V34-01 (auditoria del espacio experimental de TOXIC: que fuentes siguen sin probar)).

**Estado `AUDIT_ONLY`. Alcance: no se ejecuta ningun modelo. Se clasifica cada eje historico del registry y se identifican las fuentes sin probar usando el criterio de novedad de la directiva**

un candidato solo cuenta como nuevo si cambia al menos una de: fuente de informacion, representacion fundamental, objetivo de entrenamiento, inductive bias, forma de explotar multiples observaciones. Cambiar solo C, threshold, arboles, clausulas, bins o detalles de features NO cuenta.

## A. Tabla de frontera

| eje | experimentos | estado |
|---|---:|---|
| LINEAL | 138 | CLOSED |
| KERNEL | 5 | CLOSED |
| BOOSTING | 3 | CLOSED |
| ENSEMBLE | 73 | CLOSED |
| DEEP | 36 | CLOSED |
| EMBEDDINGS | 32 | PARTIALLY_EXPLORED |
| LABEL_NOISE | 18 | CLOSED |
| FUZZY_MATCHING | 18 | PARTIALLY_EXPLORED |
| LABEL_SPECIALIZATION | 50 | CLOSED |
| STRUCTURAL | 50 | CLOSED |
| THRESHOLD | 120 | CLOSED |
| REPRESENTACION | 107 | CLOSED |
| META | 70 | CLOSED |
| SEMI_SUPERVISED | 0 | UNEXPLORED |
| METRIC | 1 | PARTIALLY_EXPLORED |
| TUNING | 4 | PARTIALLY_EXPLORED |
| VALIDACION | 69 | CLOSED |
| AUDITORIA | 30 | CLOSED |
| PSEUDO_AUTOR | 19 | CLOSED |
| CLUSTER | 110 | CLOSED |
| FEATURES | 49 | CLOSED |
| SUBWORD | 6 | CLOSED |
| OBJECTIVE | 15 | CLOSED |
| MULTI_LABEL | 5 | CLOSED |

## B. Las ocho direcciones especificas

### 1_lineales_avanzados

- estado_general: CERRADA (17 de 24)
- LinearSVC: probado=True ids=['MOD-01_BEST', 'MOD-01_W12', 'ENS-01', 'ENS-02']
- SGD: probado=True ids=['V4-G01-G03']
- L1_Elastic: probado=False ids=[]
- FTRL: probado=False ids=[]
- PassiveAggressive: probado=True ids=['V4-G01-G03']
- NB_SVM: probado=True ids=['E2-15-NB-WORD', 'E2-15-NB-CHAR', 'E2-15-NBSVM', 'E2-15-NBSVM-SVC']

### 2_supervision_de_features

- estado: CERRADA (3 de 3)
- ids: ['E2-15-NB-WORD', 'E2-15-NB-CHAR', 'E2-15-NBSVM']
- nota: log-count ratios probado en E2-15 (negativo). class-dependent scaling y discriminative reweighting NO probados

### 3_representaciones_subword

- estado: UNEXPLORED
- ids: []
- nota: hashing trick, BM25 y binarizacion como features NO probados. char n-grams probados en V03 y E2-15

### 4_embeddings_desde_cero

- estado: CERRADA (6 de 19)
- ids: ['DL-01', 'DL-02', 'DL-03', 'EBD-00-SCAFFOLD', 'E-CORP-01', 'EBD-01', 'EBD-03', 'EBD-05']
- nota: FastText y word2vec probados (EBD, cerrado). Embeddings supervisados y sentiment embeddings NO probados

### 5_objetivos_supervisados

- estado: CERRADA (1 de 2)
- ids: ['V7-01', 'V13-00']
- nota: focal loss probado (V7-01, cerrado). class-balanced probado (V14-00, cerrado). margin/robust NO probado

### 6_semi_supervised

- estado: UNEXPLORED
- ids: []
- nota: NO probado. La directiva manda no asumir que el pseudo-labeling es legal

### 7_contrastive_metric

- estado: CERRADA (1 de 1)
- ids: ['V12-00']
- nota: NO probado. Requiere adaptacion compatible con las restricciones

### 8_multi_etiqueta

- estado: CERRADA (3 de 5)
- ids: ['B1', 'S04', 'V6-01', 'V6-02', 'V12-04']
- nota: condicionalizacion y B2 probados (V6-02, cerrado). Otras formulaciones NO probadas

## C. Top 5 de familias realmente abiertas

### 1. SGD con perdidas robustas

- mecanismo: descenso de gradiente estocastico con perdidas hinge, modified_huber o squared_hinge
- evidencia de diferencia: cambia el inductive bias: no es un solver convexo completo como liblinear, es un metodo online con penalizacion distinta. Ningun SGD aparece en el registry
- por que V03 no la captura: V03 usa liblinear, un solver dual con convergencia exacta. SGD tiene una geometria de optimizacion distinta y puede encontrar una frontera diferente con los mismos features
- experimento minimo: SGDClassifier con hinge y modified_huber, mismos features de V03, mismos pliegues
- coste: bajo, mas rapido que liblinear | riesgo overfitting: bajo, mismo numero de parametros | riesgo repeticion: bajo, no hay ningun SGD en 280 filas

### 2. Hashing trick

- mecanismo: vectorizacion por hash sin vocabulario explicito, con colisiones controladas
- evidencia de diferencia: cambia la representacion fundamental: no hay vocabulario, la dimensionalidad es fija y las colisiones son parte del metodo. NO probado en el registry
- por que V03 no la captura: V03 usa un vocabulario de 27431 columnas con min_df=2. El hashing tiene dimensionalidad fija y trata las colisiones de forma distinta, lo que puede generalizar diferente con pocos datos
- experimento minimo: HashingVectorizer con 2^18 features, mismos n-grams, mismos pliegues
- coste: bajo, sin vocabulario que almacenar | riesgo overfitting: medio, las colisiones pueden introducir ruido | riesgo repeticion: bajo, no hay ningun hashing en el registry

### 3. BM25 weighting

- mecanismo: ponderacion estilo BM25 en vez de TF-IDF sublinear
- evidencia de diferencia: cambia la representacion fundamental: BM25 normaliza por longitud del documento de forma explicita, lo que es distinto a sublinear_tf. NO probado
- por que V03 no la captura: V03 usa sublinear_tf, que es una transformacion suave. BM25 tiene una saturacion por longitud que puede comportarse diferente en textos largos, que es donde V03 falla mas
- experimento minimo: BM25Vectorizer o implementacion manual sobre los mismos n-grams, mismos pliegues
- coste: bajo | riesgo overfitting: bajo, mismo numero de features | riesgo repeticion: bajo, no hay BM25 en el registry

### 4. Embeddings supervisados desde cero

- mecanismo: entrenar embeddings sobre el corpus permitido con objetivo supervisado, no solo Skip-gram no supervisado
- evidencia de diferencia: cambia la fuente de informacion: los embeddings se entrenan con las etiquetas, no solo con la co-ocurrencia. FastText y word2vec probados fueron no supervisados
- por que V03 no la captura: V03 no usa embeddings. Un embedding supervisado puede capturar semantica de toxicidad que los n-grams no ven
- experimento minimo: entrenar embeddings con las etiquetas sobre el corpus permitido, promediar por documento, concatenar a V03
- coste: medio, requiere entrenamiento de embeddings | riesgo overfitting: medio-alto, con 1731 documentos el embedding puede memorizar | riesgo repeticion: medio, hay que verificar que no sea equivalente a EBD

### 5. Passive-Aggressive

- mecanismo: metodo online con actualizacion pasiva cuando acierta y agresiva cuando falla
- evidencia de diferencia: cambia el inductive bias: es un metodo online con margen, no un solver convexo. NO probado
- por que V03 no la captura: V03 usa liblinear. Passive-Aggressive tiene una geometria distinta y puede encontrar una frontera diferente
- experimento minimo: PassiveAggressiveClassifier, mismos features, mismos pliegues
- coste: bajo | riesgo overfitting: bajo | riesgo repeticion: bajo, no hay ningun PA en el registry

## D. Lo que este script no hace

- No ejecuta ningun modelo.
- No genera submission.
- No modifica V03.
- No abre nuevas features.

Coste: 0.4 s, memoria 14 MB, 0 modelos, 0 submissions.

