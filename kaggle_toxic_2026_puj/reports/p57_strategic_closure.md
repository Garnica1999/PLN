# P57 - Cierre estrategico y plan final de competencia

Generado por `experiments/v57_00_sintesis_estrategica.py`. Fecha 2026-09-28T03:33:59+00:00.

**0 modelos. 0 entrenamientos. 0 submissions. 0 consultas a Kaggle. 0 uso de holdout. 0 uso de test.** Esta fase es de sintesis, auditoria y estrategia. No se registro ninguna fila nueva en `experiments.csv`, porque no hubo ninguna ejecucion experimental.

## 1. Baseline congelado

```text
id            v03_lr_union_thr
repr          TF-IDF word(1,2) + char_wb(3,5), min_df=2, sublinear_tf
clasificador  LogisticRegression C=2.0, class_weight=balanced, liblinear, max_iter=1000, seed 42
umbrales      HS 0.44 | TR 0.51 | AG 0.46
corrected_cv  0.7772114525165721
LB            0.78306
sha256        7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
```

sha256 verificado en el arranque de esta fase: **True**. Intacto.

## 2. Jerarquia de magnitudes, sin mezclar metricas

Esta tabla es el resultado mas importante de P57. **Cada fila lleva su metrica, su protocolo y su poblacion, y las filas NO son intercambiables.**

| magnitud | valor | experimento | metrica | protocolo | poblacion | comparable con V03 |
|---|---:|---|---|---|---|---|
| best_architectural_CV_macro6_diagnostica | 0.8743 | S08 | macro-6, NO la metrica de la competencia | HOLD | holdout 248 | False |
| best_holdout_CV | 0.8465 | S08 (meta de 44 features) | metrica del proyecto | HOLD | holdout 248 | False |
| best_CV_protocolo_D004 | 0.8039 | S05 / v05_optuna | metrica del proyecto | D-004 | dev | False |
| best_comparable_CV_D016 | 0.7904 | V4-G01-G03 | metrica del proyecto | D-016 | dev 1483 | True |
| best_honest_CV_anidado | 0.7885 | E2-13-META-ENT-RIDGE | metrica del proyecto | D-016 | dev 1483 | True |
| V03_corrected_cv | 0.7772114525165721 | v03_lr_union_thr | metrica del proyecto | D-016 | dev 1483 | True |
| V03_historical_cv_D004 | 0.7989 | v03_lr_union_thr | metrica del proyecto | D-004 | dev | False |
| V03_holdout | 0.8285 | v03_lr_union_thr | metrica del proyecto | HOLD | holdout 248 | False |
| best_verified_LB | 0.78306 | v03_lr_union_thr | leaderboard de Kaggle | externo | test 1154 | True |
| current_champion | v03_lr_union_thr | v03_lr_union_thr | leaderboard de Kaggle | externo | test 1154 | True |
| remaining_submission_budget | limite diario exacto desconocido; 0 envios hoy; reserva propia de 3 | n/d | n/d | n/d | n/d | False |

Lo que dice esta tabla:

- El **maximo absoluto** de la columna `macro_f1` del registro es 0.8743, y **no es** una medida de la metrica de la competencia: es macro-6 sobre el holdout.
- La metrica del proyecto sobre el OOF de V03 vale 0.7772114525165721 y la macro-6 vale 0.8182213363: **+0.04101 de diferencia**, segun P45.
- `leaderboard_log.csv` registra para V03 un `local_f1` de 0.7989, que es **+0.021689** por encima del `corrected_cv` canonico, y la diferencia es exactamente el coste de medir con D-004 en vez de D-016.
- El **mejor CV comparable bajo D-016** es **0.790400** (V4-G01-G03), linea svd300_rbf. Se envio y dio LB 0.77938. **El mejor CV comparable no es el campeon.** Y es el mejor de cuatro candidatos comparados sobre el mismo OOF, luego lleva seleccion.
- El **mejor CV comparable Y anidado** es **0.788500** (E2-13-META-ENT-RIDGE), que es la cifra que un argumento de 'se ha probado mas cosas y ninguna supera a V03' tiene que citar.
- El **unico instrumento externo** es el leaderboard, y su maximo verificado es **0.78306**.

## 3. Inventario final de experimentos

`reports/p57_final_experiment_inventory.csv`, 323 filas, una por experimento del registro, con las columnas `metrica`, `protocolo`, `poblacion` y `anidado` que hacen utilizable el registro.

| clase | filas |
|---|---:|
| protocolo D-004 (mas debil, no comparable) | 58 |
| protocolo D-016 (corregido, comparable) | 102 |
| holdout de 248 filas (poblacion distinta) | 13 |
| sin clasificar | 150 |

**Ningun experimento historico se modifica.** La clasificacion se anade al lado.

## 4. Estado consolidado de familias

`reports/p57_family_status.csv`, 22 familias. Tipos de cierre: **A** resultado negativo, **B** implementacion invalida, **C** sin transferencia, **D** evidencia insuficiente, **E** prohibido o inexistente, **F** redundante.

| familia | estado | tipo | mejor resultado | LB |
|---|---|---|---|---|
| embeddings | CLOSED | E | E2-19 y V38-01: supervised embeddings CLOSED, 0.595466. From-scratch D | no enviado |
| ordinal | CLOSED | A | V4-E01-E03 TRACK E = NO_SIGNAL; V13-02 0.777746 (delta +0.000534); V14 | no enviado |
| joint_labels | CLOSED | A | 6 variantes E2-01-* (HS-TR-AG, HS-AG-TR, AG-HS-TR, con y sin estandari | no enviado |
| PLS | INVALIDATED | B | V40-01 SUPERVISED_PLS 0.763469 (OVERFIT); V43-02 resultado verdadero - | no enviado |
| VW / optimizacion sparse online | CLOSED | A | V44-01: FTRL, Coin Betting y PiSTOL sobre la representacion exacta de  | no enviado |
| contextual_selection | CLOSED | A | P0 (no hacer nada) = V03. Todas las politicas observables por debajo | no enviado |
| threshold_transfer | CLOSED | A | P55 reporto 0.7781367700. P56 con anidamiento completo da 0.7745478730 | no enviado |
| data_expansion | CLOSED | E | P47: 1731 filas oficiales, 1483 en dev y 248 en holdout. 0 filas etiqu | no enviado |
| augmentation | CLOSED | A | V6-08: espaciado, elongacion y repeticion de puntuacion. J1 (una copia | no enviado |
| retrieval | CLOSED | A | V4-F01-F04: propagacion de etiquetas descartada, techo medido NULO ant | no enviado |
| kernels | CLOSED | C | V4-X01 candidato D (svd300_rbf) corrected_cv 0.7904, 5/5 pliegues, McN | RECHAZADO |
| FM (factorization machines) | CLOSED | A | V25-01: k=4, lambda_v=1, 0.782699 bajo D-004. V6-02 Track B cerrado po | no enviado |
| Tsetlin | CLOSED | A | V26-03: 0.583397, cortado en K=200 por los criterios de la fase | no enviado |
| BM25 | CLOSED | A | V37-01: 0.7744254 | no enviado |
| hashing | CLOSED | A | V36-01: 0.748427 | no enviado |
| SGD | CLOSED | A | V35-01: hinge -0.016177, modified_huber -0.033645 | no enviado |
| label_specialization | CLOSED | C | V28-00 LABEL_SPECIALIZATION = CLOSED. v09_p5: C por etiqueta 4,0 / 2,0 | RECHAZADO |
| noise (label noise y reweighting) | CLOSED | A | V30-00 LABEL_NOISE = CLOSED. V4-B03 down_violacion LR 0.7784, IC95 [-0 | no enviado |
| fuzzy_matching | CLOSED | A | V29-03 REJECTED, FUZZY_MATCHING = INFORMATIONAL_ONLY | no enviado |
| ensembles | CLOSED | C | ENS-02 anidado 0.791900, descartada por no superar MOD-03b en +0.005.  | a la baja |
| stacking y meta-modelos | CLOSED | C | S08 meta holdout 0.8465 frente a 0.8285 de V03, delta +0.0180. E2-13-M | a la baja |
| structural_features | CLOSED | A | V15-01: 0.775860, NO ADOPTADA. V14-00 0.660630. V32-05 url_length: cor | no enviado |

### Las cuatro familias con veredicto de tipo C (sin transferencia)

| familia | envios | LB |
|---|---:|---|
| kernels | 1 | 0.77938, RECHAZADO |
| label_specialization | 1 | 0.77781, RECHAZADO |
| ensembles | 3 | 0.7751, 0.7747, 0.77122, los tres a la baja |
| stacking / meta-modelos | 5 | 0.78163, 0.77895, 0.77978, 0.78150, 0.77663, los cinco a la baja |

**Diez envios de candidatos con mecanismo nuevo. Los diez transferieron a la baja.** Este es el hecho mas firme del proyecto, y es lo que gobierna la estrategia final.

### La distincion que la directiva pide (A frente a no probado)

| tipo | familias |
|---|---|
| A | ordinal, joint_labels, VW / optimizacion sparse online, contextual_selection, threshold_transfer, augmentation, retrieval, FM (factorization machines), Tsetlin, BM25, hashing, SGD, noise (label noise y reweighting), fuzzy_matching, structural_features |
| B | PLS |
| C | kernels, label_specialization, ensembles, stacking y meta-modelos |
| E | embeddings, data_expansion |

**Ninguna familia queda como D (evidencia insuficiente)**, salvo dos casos que se senalan explicitamente y que **no** se cuentan como cerradas por resultado:

1. **PLS** es tipo **B**: V42-01 y V43-01 se invalidaron por implementacion, y su numero no es evidencia. El resultado verdadero lo produjo V43-02 como verificacion y es negativo (-0,013742). La hipotesis no se rechaza, el experimento.
2. Las lineas de estructura de etiquetas de V13-02 y V14-00 se cerraron **por condicion de parada**, con 0 de 4 experimentos ejecutados. Ahi **no se puede decir 'no funciono'**, porque no llego a probarse. Se registra como pregunta abierta, no como cierre.

## 5. V03 y sus variantes

| variante | umbrales | honest_CV | LB | estado |
|---|---|---:|---:|---|
| **V03 congelado** | 0.44/0.51/0.46 | 0.7772114525165721 | 0.78306 | CAMPEON |
| P55, umbrales anidados simples | reestimados | 0.7781367700 | n/d | superseded por P56 |
| P56 T1, mediana aplicada globalmente | 0.41/0.46/0.45 | 0.7798514633 | n/d | optimista en +0.0053 |
| P56 T1, umbral por pliegue exterior | por pliegue | 0.7745478730 | n/d | **NOT_CONFIRMED** |
| v02 (mismo modelo, umbral 0.5) | 0.50/0.50/0.50 | n/d | 0.77993 | REJECTED |
| v04 ENS-04 (ensemble LR+SVC) | medianos por pliegue | n/d | 0.7751 | REJECTED |
| v16_ch7 C=1.5 | 0.44/0.51/0.46 | 0.776653 | 0.77296 | REJECTED |
| v16_ch8 C=2.5 | 0.44/0.51/0.46 | 0.776752 | 0.77707 | REJECTED |
| v16_ch9 min_df=1 | 0.44/0.51/0.46 | 0.765532 | 0.76978 | REJECTED |
| v16_ch1 tasa interpolada | quantile | n/d | 0.77828 | REJECTED |
| v16_ch5 tasa optima absoluta | 0.410/0.475/0.450 | n/d | 0.77495 | REJECTED |
| v32_05 url_length | 0.44/0.51/0.46 | identica a V03 | 0.78306 | delta 0, WEAK_TRANSFER |

Las tres dimensiones de un solo parametro tienen su optimo en V03: C (1.0 -> 0.77227, **2.0 -> 0.78306**, 3.26 -> 0.77602), min_df (1 -> 0.76978, **2 -> 0.78306**, 5 -> 0.77034) y tasa predicha (0.3385 -> 0.77993, **0.3645 -> 0.78306**, 0.3836 -> 0.77828, 0.3980 -> 0.77495). El maximo esta en V03 en las tres, y la respuesta a la tasa es monotona a ambos lados.

## 6. Cierre de thresholds (P55 / P56)

```text
P55 reported          0.7781367700
P56 global            0.7798514633   (+0.0026400107, 4/5 pliegues)
P56 honesto           0.7745478730   (-0.0026635795)
optimismo de la cifra global   +0.0053035902
V03                   0.7772114525
THRESHOLD_TRANSFER = CLOSED
```

Los umbrales de V03 caen **dentro** del rango de las cinco estimaciones interiores en las tres etiquetas. El defecto procedimental de P55 sigue siendo un hecho historico real, pero su magnitud con el procedimiento correcto es nula o negativa. La familia no se reabre cambiando la mediana por la media, ni la rejilla, ni offsets pequenos.

## 7. Cierre de los canales (P49 a P53)

```text
V03 errores            757 celdas (4449 totales)
V03 correctos          3692
rescuable              715
no rescatable          42 celdas en 33 tweets
true oracle            0.9870178614  (+0.2098064089)
mejor politica observable  0.7753804469  (-0.0018310056)
CONTEXTUAL_SELECTION = CLOSED
```

**La formulacion correcta, y es importante:**

- **Lo que se SOSTIENE:** la senal existe y es enorme. El techo del oracel es +0.2098064089, casi 70 veces el margen de adopcion de 0,003. Las 715 celdas son alcanzables en principio.
- **Lo que NO se SOSTIENE:** que exista una politica observable que convierta esa senal en mejora neta. P53 midio que los 20 canales que discrepan proponen `1 - v03_pred` en las 3896 celdas con discrepancia, sin una sola excepcion, luego el espacio de acciones real cae de 21 a 2 y toda politica esta por debajo de V03.

Decir 'la senal existe pero ningun selector probado la convirtio en mejora neta positiva' es correcto. Decir 'no hay senal' seria falso, y el registro no lo dice.

## 8. Error residual

```text
42 celdas no rescatables = 0.94 por ciento de las 4449
soporte efectivo: 33 tweets, no 42
```

P54: FP 22 / FN 20, 85,7 por ciento de margen alto, 36 de 42 de alta confianza. HS 2,85 por ciento, TR 7,32 por ciento, AG 7,58 por ciento de los errores. El lexico de insulto discrimina 1,25 veces contra A, luego **no discrimina**.

**No se exagera esta poblacion.** 42 celdas en 33 tweets es el techo de lo que queda, y aun con un oracel perfecto sobre el 0,95 por ciento del espacio el movimiento del macro-F1 ronda +0,009, por debajo de la banda de particion de 0.017184 de D-070. La formulacion correcta es **'no hay ningun mecanismo nuevo con soporte suficiente y valor esperado material'**, no 'es imposible mejorar'.

## 9. Auditoria de datos

| magnitud | valor |
|---|---:|
| train oficial etiquetado | 1731 |
| desarrollo | 1483 |
| holdout | 248 |
| test oficial | 1154 |

```text
DATA_EXPANSION = CLOSED
TRAIN_AUGMENTATION = CLOSED
```

Las 1731 filas estan todas asignadas: 1483 en dev y 248 en holdout. Quedan **0** filas oficiales etiquetadas sin usar. Los corpus externos existen (42808 documentos, 95.0M tokens) pero no estan etiquetados, el manifiesto declara contaminacion cero con `toxic_test.csv`, y su unico uso permitido es entrenar embeddings, familia ya cerrada.

## 10. Submissions

| magnitud | valor |
|---|---:|
| envios historicos totales | **23** |
| con LB verificado en el log | 22 |
| envio real no registrado en el log | 1 |
| ficheros de submission en disco | 37 |
| generados y nunca enviados | 14 |
| maximo LB verificado | **0.78306** |
| minimo LB registrado | 0.51494 (plantilla todo-uno) |
| envios hoy, 2026-09-28 | 0 |

**El envio que falta en el log.** V33-03 documento un envio real y aceptado: ID `56608788`, fecha 2026-09-27, fichero `toxic_submission_v32_05_url_length.csv`, LB 0.78306, delta +0.00000. Su anotacion de presupuesto decia 'V33-03 anoto '2 submissions restantes hoy' el 2026-09-27, dia en el que se cuentan 7 envios. Eso exige un limite diario de al menos 9, pero la cifra EXACTA no esta registrada en ningun sitio local.'

Envios por dia, con el envio no registrado sumado:

| dia | envios |
|---|---:|
| 2026-09-22 | 6 |
| 2026-09-23 | 2 |
| 2026-09-25 | 4 |
| 2026-09-26 | 4 |
| 2026-09-27 | 7 |

### Presupuesto restante

```text
limite diario EXACTO        False   (no consta en ningun registro local)
envios hoy                  0
reserva propia del proyecto  3
```

**No se consulta Kaggle.** V33-03 anoto '2 submissions restantes hoy' el 2026-09-27, dia en el que se cuentan 7 envios. Eso exige un limite diario de al menos 9, pero la cifra EXACTA no esta registrada en ningun sitio local. La reserva de 3 es una autolimitacion del proyecto, no un limite de la competencia.

### Artefactos que no deben volver a usarse

| artefacto | clase | motivo |
|---|---|---|
| `toxic_submission_v17_prior_shift_em.csv` | QUARANTINADA, transductiva | consume el EM de prevalencia sobre el conjunto de test |
| `toxic_submission_v18_near_dup_propagacion.csv` | QUARENTINADA, transductiva | propaga etiquetas usando near-duplicados del test; ademas cambia 0 celdas |
| `toxic_submission_v16_ch2_prior_shift_em.csv` | CUARENTENADA, transductiva | mismo mecanismo que v17, mismo motivo. No se envio, y por coherencia no debe enviarse |
| `toxic_submission_v12_descartado_meta36_downweight.csv` | RECHAZADA | holdout 0.8163 contra 0.8285 de V03, delta -0.0122. Consume la quinta y ultima evaluacion de holdout |
| `toxic_submission_v20_03_reproduccion_v03.csv` | INVALIDA | es una reproduccion de V03, no una candidata |
| `toxic_submission_v32_05_url_length.csv` | GASTADA, delta 0 | envio real, LB identico a V03 porque el fichero es identico |

Ninguna se modifica. Ninguna se vuelve a enviar.

## 11. Preguntas abiertas con mecanismo

`reports/p57_open_questions.csv`, 5 filas. Cada una declara un mecanismo, no un 'no se ha probado'.

| pregunta | probada | valor esperado |
|---|---|---|
| La brecha OOF -> LB de 0,00584855 de V03 no tiene mecanismo identificado | PARCIAL: el shift se midio y se co | El unico eje del proyecto con un contraste cuasi experimen |
| La inversion del ranking local al LB en la familia de meta no tiene causa iden | SI, extensamente: 4 envios de meta | ALTO conceptualmente: identificarla deciria que magnitud l |
| La hipotesis de ambiguedad de etiqueta en HS quedo registrada sin medir | NO. Registrada en V28-03 sin ejecu | CUANTIFICADO Y BAJO |
| La sensibilidad del LB esta limitada por el numero de filas del test | SI. V8-02 adopto un filtro de dete | Restriccion permanente |
| Ninguna linea de estructura de etiquetas se ejecuto porque se cerro por condic | NO, por condicion de parada | BAJO, pero por un motivo que conviene precisar: V13-01, qu |

**Ninguna cumple hoy el umbral de reapertura** de la directiva, que exige las seis condiciones a la vez. Se dejan documentadas porque su ausencia de evidencia es especifica y esta medida, no vaga.

## 12. Candidatos historicos

`reports/p57_candidate_matrix.csv`, 7 artefactos. **Tabla descriptiva, sin ranking y sin puntuacion.**

| artefacto | mecanismo | honest_CV | delta | LB | verificado | disponible |
|---|---|---:|---:|---:|---|---|
| v23_01 svd300_rbf (V4-X01 candidato D) | SVD 300 + SVC kernel RBF con calibracion sig | 0.7904 | 0.013189 | 0.77938 | SI, RECHAZADO | SI, en disco, ya gasta |
| v08_meta_v2 (meta de 44 features, Ridge alph | stacking sobre 4 bases TF-IDF mas 8 rasgos e | 0.7874 | 0.010189 | 0.78163 | SI, RECHAZADO | SI, en disco, ya gasta |
| v20_01 regla HS por interaccion de scores (V | LR C=1.0 sobre [logit(HS), logit(HS)*logit(T | 0.7882 | 0.010989 | 0.77663 | SI, RECHAZADO. Ade | SI, en disco (toxic_su |
| E2-13-META-ENT-RIDGE (meta + entropias) | Ridge sobre el meta ampliado con entropias d | 0.7885 | 0.011289 | n/d | NO | NO |
| V03 con umbrales anidados de P56 (T1 = 0,41/ | umbral por etiqueta como mediana de cinco es | 0.774547873 | -0.0026635795165720344 | n/d | NO | NO |
| v25_01 FM k=4 | factorization machines sobre las cuatro base | 0.782699 | 0.005487 | n/d | NO | NO |
| v16_ch2 correccion de prior con EM | desplazamiento del logit con los factores po | n/d (validacion anidada de V9-02 dio +0.000574) | 0.000574 | n/d | NO | SI, en disco, pero CUA |

### Aplicando los cinco criterios simultaneos de la directiva

```text
legal            + reproducible     + sin leakage
no rechazado ya  + mecanismo claro
```

**Resultado: NINGUN artefacto cumple los cinco a la vez.**

| artefacto | que le falta |
|---|---|
| v23_01 svd300_rbf | ya rechazado: enviado, LB 0.77938 |
| v08_meta_v2 | ya rechazado: enviado, LB 0.78163 |
| v20_01 meta 36 features | ya rechazado dos veces: por holdout y por LB 0.77663 |
| E2-13-META-ENT-RIDGE | sin leakage y legal, pero cerrado por sparsity y nunca verificado fuera |
| umbrales anidados de P56 | veredicto NOT_CONFIRMED, delta honesto negativo |
| v16_ch2 prior shift EM | transductivo: usa informacion del conjunto de test |

Es un resultado, no una omision. Los tres artefactos con mejor evidencia interna tienen **veredicto externo medido y contrario**, y el unico con delta honesto positivo y sin veredicto externo fue cerrado por una razon de mecanismo, no de metrica.

## 13. Inconsistencias documentales, seccion 25 de la directiva

7 hallazgos, clasificados sin corregir la historia experimental.

| severidad | donde | inconsistencias |
|---|---|---|
| **CRITICAL** | experiments.csv, columnas `f1` y `macro_f1` | Las dos columnas mezclan al menos TRES metricas y TRES protocolos. `macro_f1` contiene m |
| **CRITICAL** | submissions/leaderboard_log.csv frente a experiments | V33-03 documento un envio REAL a Kaggle: ID 56608788, fecha 2026-09-27, fichero toxic_su |
| **IMPORTANT** | experiments.csv, fila V43-01 | La columna `question` dice 'PLS Regression como proyeccion supervisada document x term', |
| **IMPORTANT** | leaderboard_log.csv, filas 56597267 y 56599802 | La columna `archivo` registra el nombre sin el prefijo `submissions/`, mientras que las  |
| **IMPORTANT** | cifras que circulan por el repositorio | P47 reconcilio dos cifras que no cuadraban: 'holdout = 519' y 'unas 44.000 filas no etiq |
| **MINOR** | CHAMPION.md, seccion 4 | La tabla de candidatas de V16 titula una columna 'LB combinado' que contiene PREDICCIONE |
| **MINOR** | CHAMPION.md, seccion 1 | El bloque del campeon declara 'max_iter 2000', mientras que CHAMPION.md y el manifiesto  |

### 1. [CRITICAL] experiments.csv, columnas `f1` y `macro_f1`

- **Inconsistencia:** Las dos columnas mezclan al menos TRES metricas y TRES protocolos. `macro_f1` contiene macro-6 (diagnostica) en la era V2/E2/MOD/TH/ENS y la metrica del proyecto en la era V10+. `f1` contiene holdout de 248 filas en S08, protocolo D-004 en la era antigua y D-016 en la era corregida.
- **Evidencia:** P45 sobre el OOF de V03: metrica del proyecto 0.7772114525 y macro-6 0.8182213363, diferencia +0.04101. MOD-03b_BEST: mf 0.834900 contra f1 0.792700, diferencia +0.042200. El maximo de `macro_f1` en todo el registro es 0.8743 (S08), que es macro-6 sobre holdout.
- **Consecuencia:** Cualquier tabla de 'mejor CV' construida tomando el maximo de esas columnas es FALSA por mas de 0,04, y mixing protocolos anade hasta 0,024 mas. Este script NO construye tal tabla: etiqueta cada fila con metrica, protocolo y poblacion, y solo compara dentro de un mismo protocolo.
- **Correccion:** No se corrige el registro historico. Se anade la clasificacion en `reports/p57_final_experiment_inventory.csv`, que es la forma de conservar el significado original y hacer la columna utilizable.

### 2. [CRITICAL] submissions/leaderboard_log.csv frente a experiments.csv

- **Inconsistencia:** V33-03 documento un envio REAL a Kaggle: ID 56608788, fecha 2026-09-27, fichero toxic_submission_v32_05_url_length.csv, LB 0.78306, y anota '2 submissions restantes hoy'. Esa fila NO esta en leaderboard_log.csv.
- **Evidencia:** experiments.csv fila V33-03, campo `result`. El log tiene 22 filas y ninguna corresponde a v32_05_url_length. Hay 37 ficheros de submission en disco y 15 sin fila en el log, de los cuales solo ESTE se envio.
- **Consecuencia:** El total historico de envios es 23, no 22. PROJECT_STATE.md afirma '0 de 22 submissions superaron V03', y CHAMPION.md auditado en P46 afirma lo mismo. La afirmacion sigue siendo cierta en su sustancia, porque 0 de 23 superaron 0.78306, pero el recuento es erroneo.
- **Correccion:** Se anade la fila al informe de P57 con su ID y su LB, y se deja el log intacto. No se modifica un registro historico de measurements; se documenta la discrepancia y se da la cifra correcta.

### 3. [IMPORTANT] experiments.csv, fila V43-01

- **Inconsistencia:** La columna `question` dice 'PLS Regression como proyeccion supervisada document x term', pero el `script` que registra es experiments/v43_01_vw_optimizers.py, cuyo docstring es sobre Vowpal Wabbit con FTRL, Coin Betting y PiSTOL. Ademas la `decision` habla de la API de VW.
- **Evidencia:** docstring de experiments/v43_01_vw_optimizers.py frente a la fila V43-01 del registro
- **Consecuencia:** El texto de la pregunta no corresponde al experimento. Quien lea el registro creera que V43-01 evalua PLS, cuando lo que se intento fue FTRL y quedo INVALIDADO por la API de VW. La familia PLS se cerro en V43-02 y V44-02.
- **Correccion:** Se documenta en P57. NO se edita la fila historica. El sentido original se conserva: V43-01 es el intento de FTRL, y su invalidacion es la que abre V44-01.

### 4. [IMPORTANT] leaderboard_log.csv, filas 56597267 y 56599802

- **Inconsistencia:** La columna `archivo` registra el nombre sin el prefijo `submissions/`, mientras que las otras 20 filas lo incluyen. Los ficheros existen en `submissions/`.
- **Evidencia:** log dice 'toxic_submission_v20_01_...' y 'toxic_submission_v23_01_...'; ambos existen como submissions/...
- **Consecuencia:** Un lector que resuelva la ruta literalmente concluiria que faltan dos artefactos. No falta ninguno.
- **Correccion:** Se documenta. No se modifica el log, porque las dos filas tienen un ID numerico de Kaggle y su contenido es fiable.

### 5. [IMPORTANT] cifras que circulan por el repositorio

- **Inconsistencia:** P47 reconcilio dos cifras que no cuadraban: 'holdout = 519' y 'unas 44.000 filas no etiquetadas'.
- **Evidencia:** 519 es el numero de positivos de TR en train, no el tamano del holdout, que es 248. Y no existe ninguna fuente de 44.000 filas: los corpus externos se miden en tokens, y el 44 es el numero de features del meta-modelo E2-02.
- **Consecuencia:** Ambas cifras circulaban como hechos del inventario de datos y ambas son falsas.
- **Correccion:** Ya corregido en P47 (reports/p47_data_inventory.json, campo reconciliacion_de_cifras). Se confirma aqui.

### 6. [MINOR] CHAMPION.md, seccion 4

- **Inconsistencia:** La tabla de candidatas de V16 titula una columna 'LB combinado' que contiene PREDICCIONES, no mediciones. El propio CHAMPION.md lo advierte en un bloque anadido por P46.
- **Evidencia:** valores 0.78535 (ch1) y 0.78650 (ch5) frente a sus LB medidos reales 0.77828 y 0.77495
- **Consecuencia:** Las dos candidatas que parecian ganadoras por la prediccion perdieron por 0.007 y por 0.012 respectivamente. Leer la columna sin el aviso seria el error de lectura mas caro del proyecto.
- **Correccion:** El aviso existe y es correcto. Se mantiene la tabla intacta como registro de lo que se predijo, con el aviso encima. Sin cambio.

### 7. [MINOR] CHAMPION.md, seccion 1

- **Inconsistencia:** El bloque del campeon declara 'max_iter 2000', mientras que CHAMPION.md y el manifiesto declaran max_iter 1000.
- **Evidencia:** CHAMPION.md linea del bloque Version v03 dice max_iter=1000; el bloque 'Champion conservado' del final dice max_iter 2000
- **Consecuencia:** Estandar de liblinear: el valor no cambia la solucion. Es una discrepancia documental, no un riesgo numerico.
- **Correccion:** Se documenta. El pipeline congelado experiments/baseline_v03_frozen/pipeline_v03.json es la fuente de verdad y no se modifica.

## 14. Integridad del notebook

- celdas: **210**
- seccion P57 anadida como seccion 53, celda de codigo con returncode 0 y cifras leidas de los artefactos, no escritas a mano
- sin experimentacion nueva

## 15. Formulacion final obligatoria

> V03 remains the best verified leaderboard result and the frozen architectural baseline. The investigated legal mechanisms did not produce a robust transferable improvement, and no additional mechanism with sufficient evidence has currently been identified.

Y la distincion que hay que no perder:

```text
best verified result  !=  proof of global optimum
```

- Lo verificado: **0.78306** es el maximo del leaderboard en 23 envios, y 0.7772114525165721 es el CV corregido de la configuracion congelada.
- Lo que NO se afirma: que V03 sea globalmente optimo. La brecha OOF->LB de 0.00584855 sigue sin mecanismo identificado, y el techo del oracel de +0.2098064089 no se ha alcanzado ni se ha descartado como alcanzable.
- El hecho de que ningun candidato con mejor CV local haya transferido es evidencia en contra de la overlapped optimizacion local, y a favor de V03 como eleccion. **No es evidencia de que no exista una mejor.**

## 16. Las tres estrategias, sin elegir

| estrategia | evidencia a favor | evidencia en contra | riesgo | coste |
|---|---|---|---|---|
| **A. cerrar con V03** | 0 de 23 envios lo superaron; los 10 candidatos con mecanismo nuevo transfirieron a la baja; las 22 familias estan cerradas | el techo del oracel es +0.2098064089 y la brecha OOF-LB sigue sin explicar | bajo: 0.78306 queda registrado y el leaderboard conserva la mejor puntuacion | cero |
| **B. usar un candidato historico** | E2-13 tiene +0,0113 de delta honesto anidado y ninguna contradiccion externa | los tres con delta comparable tienen veredicto externo contrario; E2-13 se cerro por sparsity, no por resultado | alto: gasta presupuesto en repetir un patron de 10/10 | 1 submission |
| **C. ultima investigacion dirigida** | la brecha OOF-LB y la asimetria dev-holdout no estan caracterizadas | ninguna hipotesis pendiente cumple las seis condiciones de reapertura; la banda de particion de 0,017 supera cualquier delta local alcanzable | medio: puede no producir nada y consume presupuesto de submissions | alto, no medido |

**La decision competitiva queda pendiente de una instruccion posterior.** P57 no la toma.

## 17. Si se autorizara una submission

No se recomienda ninguna. Si se autorizara, el unico formato admisible es:

```text
un candidato congelado
una sola submission
no una busqueda interactiva
```

Y dado el resultado de la seccion 12, **no hay candidato congelado que cumpla los cinco criterios**. En consecuencia, bajo el estado actual del proyecto, la opcion B no es ejecutable sin una excepcion explicita.

## 18. Conclusion

V03 permanece como **mejor resultado verificado** del leaderboard (0.78306) y como **baseline arquitectonico congelado** (0.7772114525165721), con sha256 verificado. 23 envios historicos, ninguno lo supero. Las familias investigadas no produjeron una mejora robusta y transferible.

Lo que P57 anade y que las fases anteriores no Tenian formulado:

1. **La jerarquia de magnitudes, con metricas y protocolos separados.** La columna `macro_f1` del registro no es una metrica, y su maximo 0.8743 no es una medida de la competencia.
2. **Que el mejor CV comparable (0.790400, con seleccion) y el mejor anidado (0.788500) NO son el campeon, y que el primero dio LB 0.77938.** El patron de 10 envios de candidatos a la baja es la evidencia mas firme del proyecto.
3. **Que un envio real no estaba en el log**, y que el total real es 23 y no 22.
4. **Que PLS es un cierre de tipo B, no A**: su numero nunca fue evidencia, y el resultado verdadero llego despues como verificacion.
5. **Que ninguna de las lineas de estructura de etiquetas se ejecuto**, porque se cerraron por condicion de parada. Es el unico sitio del registro donde no cabe decir 'no funciono'.

*Seccion generada por `experiments/v57_00_sintesis_estrategica.py`. Sin push: la rama sigue local.*
