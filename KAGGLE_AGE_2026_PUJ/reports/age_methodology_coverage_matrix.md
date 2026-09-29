# AGE - matriz de cobertura metodologica

**Baseline:** V30 = 0.6134471, leaderboard 0.63302.

## Como se construyo

Los 22 mecanismos se buscaron en el codigo por **operacion**, no por nombre de libreria: `class_weight` aparece en 8 scripts y `LGBMClassifier` en 27, pero lo que se registra es que el mecanismo se haya **ejecutado**, no que la libreria este instalada.

## Resultado

| Estado | Cuantos |
|---|---|
| USED | 23 |
| CLOSED | 9 |
| NOT USED | 5 |
| INVALIDATED | 1 |
| NOT_ACTIONABLE | 1 |

## La matriz completa

| ID | Familia | Mecanismo | Estado | Evidencia | Modo de fallo | Novedad |
|---|---|---|---|---|---|---|
| R01 | representacion | TF-IDF de palabra (1-2) | **USED** | E-001 y siguientes; 36 scripts con TfidfVectorizer | - | n/a |
| R02 | representacion | TF-IDF char_wb (3-5) | **USED** | E-003b, 34 scripts con char_wb; fue el mejor bloque aislado de la fase temprana | - | n/a |
| R03 | representacion | skipgrams de palabra | **USED** | 4 scripts | sin aporte | duplicado |
| R04 | representacion | canales estructurales (10 de V31-03) | **USED** | V31-03: mejor canal +0,006820 con 5/5 folds sobre el texto solo | la senal es real pero V30 la absorbe: V31-06 +0,0002948 sobre el contr | duplicado |
| R05 | representacion | bloque estructurado denso de 279 features | **USED** | V20 y todo el meta-espacio posterior | - | n/a |
| R06 | representacion | SVD / reduccion de dimensionalidad | **USED** | 6 scripts, E-019 y E-025/E-026 | sin aporte sobre el TF-IDF crudo | duplicado |
| R07 | representacion | canales de posicion y forma de palabra | **CLOSED** | V31-03: -0,002303 y -0,002984, 0/5 folds | negativos en los cinco folds | duplicado |
| R08 | representacion | canales de URL (presencia, tipo, posicion) | **CLOSED** | V31-02 +0,000738 (3/5); V31-03 +0,001894 en banco, nada en integracion | la URL ya se leia: enmascararla cuesta -0,006969 | duplicado |
| M01 | familia de modelo | LogisticRegression | **USED** | 71 scripts | - | n/a |
| M02 | familia de modelo | LinearSVC | **USED** | 61 scripts | - | n/a |
| M03 | familia de modelo | SGDClassifier | **USED** | 7 scripts | - | n/a |
| M04 | familia de modelo | Naive Bayes (y NB-SVM) | **USED** | 14 scripts | - | n/a |
| M05 | familia de modelo | Ridge | **USED** | 17 scripts | - | n/a |
| M06 | familia de modelo | MLP densa | **USED** | 10 scripts, E-019 y V29 | inferior a LightGBM en el meta-espacio | duplicado |
| M07 | familia de modelo | LightGBM | **USED** | 27 scripts; V29 +0,002392 y V30 +0,000069 de refinamiento | la capacidad del meta-arbol quedo agotada: la region estaba encontrada | duplicado |
| M08 | familia de modelo | ExtraTrees / RandomForest | **USED** | V20-F1/F2, 1-2 scripts | sin aporte sobre LightGBM | duplicado |
| M09 | familia de modelo | k-NN / NearestNeighbors | **USED** | 10 scripts | sin aporte | duplicado |
| M10 | familia de modelo | prototipos / centroides | **USED** | 15 scripts, E-021 experimento_e021_ordinal_centroides | sin aporte | duplicado |
| C01 | combinacion | stacking / blending | **CLOSED** | H06 y H04; 12 scripts; V28-04 lo declaro equivalente a H04+H06 | equivalencia funcional con la ponderacion simple | duplicado |
| C02 | combinacion | ensembles de modelos historicos | **CLOSED** | V29: 0 de 47 combinaciones con rescate neto positivo | diversidad sin informacion nueva | duplicado |
| C03 | combinacion | correccion del score (global, local, pendiente) | **CLOSED** | P33, P34, G12: -0,0002152, -0,0002039 y +0,0003018 sin superar a la constante | el control de ruido aleatorio gana a la senal real en los tres casos | duplicado |
| C04 | combinacion | bagging del meta-modelo | **CLOSED** | V28-02: -0,000312 | sin aporte | duplicado |
| O01 | objetivo | multiclass cross-entropy (multi_logloss) | **USED** | 95 scripts | - | n/a |
| O02 | objetivo | cost-sensitive / class_weight | **USED** | V18-P1 peso de clases, 8 scripts con class_weight o sample_weight | sin aporte | duplicado |
| O03 | objetivo | ranking objective | **NOT USED** | 0 scripts, 0 menciones en el historial | - | **descartado por irrelevancia** |
| O04 | objetivo | focal loss | **NOT USED** | 0 scripts, 0 menciones en el historial | - | duplicado con indicacion negativa |
| O05 | objetivo | label smoothing | **NOT USED** | 0 scripts, 0 menciones en el historial | - | duplicado con indicacion negativa |
| E01 | muestreo | row weighting explicito (sample_weight) | **USED** | V18-P1, V28-02 | sin aporte | duplicado |
| E02 | muestreo | balanced sampling | **USED** | 2 scripts | sin aporte | duplicado |
| E03 | muestreo | hard example mining | **NOT USED** | 0 scripts, 0 menciones | - | duplicado con indicacion negativa |
| V01 | varianza | bootstrap de grupos | **NOT USED** | 0 scripts | - | duplicado |
| V02 | varianza | seed ensemble | **USED** | 1 script | sin aporte | duplicado |
| L01 | estructura de etiquetas | ordinal (CORAL y variantes) | **CLOSED** | H10, D-058; 47 menciones de ordinal en el historial; V22-V23 | sin aporte sobre el multinomial | duplicado |
| L02 | estructura de etiquetas | pairwise / descomposicion | **INVALIDATED** | V28-03: -0,067553, 0/5 folds, neto -10.996 | sustituir el multiclase por binarios destruye la estructura de la clas | duplicado |
| L03 | estructura de etiquetas | classifier chain | **USED** | 2 scripts | sin aporte | duplicado |
| L04 | estructura de etiquetas | calibracion / post-procesado de probabilidad | **CLOSED** | PH32 y PH33: +0,0002 y ninguna de 5 variantes positiva; P33 mostro que el intercepto se elige 0 | es efecto de calibracion, no de representacion | duplicado |
| S01 | estructura local | cluster conditioning | **USED** | 10 scripts, V13 y V31 | sin aporte | duplicado |
| S02 | estructura local | seleccion de features supervisada | **CLOSED** | V25, D-060: -0,000629 | sobreajuste a la particion | duplicado |
| S03 | estructura local | entrada no usada (G14) | **NOT_ACTIONABLE** | G14: el id es un indice de fila, 6 digitos, 100% de los ids del test reaparecen en train | no hay ningun campo de entrada legal sin usar | duplicado |

## Los mecanismos sin usar, y por que se cierran

De 22, cinco no aparecen en el codigo. No se descartan por inercia: cada uno se contrasta con la medicion que ya existe.

**O03 - ranking objective** (**descartado por irrelevancia**)

la metrica de la competicion es macro-F1 sobre el argmax. Un objetivo de ranking no mueve la metrica evaluada: cambiaria el objetivo sin cambiar el criterio

**O04 - focal loss** (duplicado con indicacion negativa)

es reponderacion de ejemplos con modulacion por confianza. `class_weight` ya se probo (V18-P1) y la modulacion apunta a los ejemplos dificiles, que son la poblacion A: 27.430 filas donde NINGUNA de las 26 fuentes tiene senal. Reponderar ahi es entrenar sobre ruido

**O05 - label smoothing** (duplicado con indicacion negativa)

actua sobre la zona de frontera, y P34 midio que ahi el ruido aleatorio gana a la senal real (real - ruido = -0,0000797)

**E03 - hard example mining** (duplicado con indicacion negativa)

el conjunto dificil ES la poblacion A, donde ninguna de las 26 fuentes tiene senal. Minar ese conjunto es exactamente entrenar sobre ruido

**V01 - bootstrap de grupos** (duplicado)

es reduccion de varianza con otra unidad de remuestreo. V28-02 cerro el bagging del meta-modelo en -0,000312, y cambiar la unidad de remuestreo no cambia el mecanismo
