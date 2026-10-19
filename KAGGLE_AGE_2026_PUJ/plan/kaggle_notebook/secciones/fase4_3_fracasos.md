## 4.3 Fracasos seleccionados [RE/DOC]

Esta sección documenta ocho fracasos seleccionados del proyecto: el MLP denso, CatBoost, FTRL-Proximal, CORAL, el modelado ordinal, los especialistas de frontera, los canales lingüísticos y la selección supervisada de features. Cada subsección declara **[RE]** si el experimento es re-ejecutable en minutos con insumos que el propio notebook reconstruye, o **[DOC]** si exige artefactos congelados (la matriz de 279 features, los OOF de V30) o procesos de horas. En todos los casos se documenta la hipótesis, el resultado exacto con su fuente y la razón del cierre: los fracasos se conservan como evidencia y no se esconden.

### 4.3.1 [RE] MLP denso (E-019/E-020)

Se entrenaron dos MLP en CPU, porque la GPU estaba ocupada por una carga ajena: un MLP(128.64) sobre las 35 features densas y un MLP(256.128) sobre densas más SVD(64) del espacio de caracteres; después se midió su aporte como miembro del ensamble. La hipótesis era que un modelo no lineal sobre densas podía competir con los lineales dispersos y que la SVD comprimiría sin perder señal.

El resultado fue doble. Como clasificadores quedaron muy por debajo: 0.3381 ± 0.0040 (peor que el LightGBM denso, 0.3567) y 0.4071 ± 0.0060, con una SVD que explica solo el 7.8 % de la varianza. Como miembro del ensamble, en cambio, el MLP llevó el meta-modelo de 0.5784 a 0.5851 (+0.0066), el mayor aporte individual desde el canal de menciones. La decisión D-015 fue precisa: el deep learning no se usa como modelo principal, pero sí como miembro por decorrelación de errores; en la era V19 el MLP(64) fue un tercio del blend (LogReg + MLP + árbol). La subsección se marca [RE] porque reconstruir el bloque de 35 densas y reentrenar cuesta unos 5-6 minutos de CPU; sin esa reconstrucción, el resultado es documentable (`experiments.csv:29-31`; `DECISIONS.md:265-281`, D-015; `RUNLOG.md:207-215`).

### 4.3.2 [DOC] CatBoost (V32-E2)

El ciclo 2 de la reapertura V32 probó CatBoost MultiClass (Prokhorenkova, Gusev, Vorobev, Dorogush y Gulin, «CatBoost: unbiased boosting with categorical features», NeurIPS 2018) sobre la matriz 279 congelada (cap de 3000 iteraciones, lr 0.03, depth 6, l2 3.0, early stopping 150), con un gemelo LightGBM afín (A2, control decisional) y etiquetas barajadas (A3, control negativo). La hipótesis era que el ordered boosting aportaría información distinta del LightGBM afilado del pool. El pre-registro exigió una enmienda previa: la métrica de evaluación se fijó como TotalF1:average=Macro, porque el default Weighted de catboost 1.2.10 no coincide con el macro-F1.

El resultado fue negativo en los tres brazos: A1 combinado 0.6148474, delta -0.0011258 (1 de 5 folds; IC95 [-0.0019378, -0.0003467]); A2 -0.0012871 y A3 -0.0012138. La comparación A1 contra A2 no fue favorable (+0.00017; IC95 contiene el cero; p+ = 0.76), y el control barajado rindió casi como A1 y por encima de A2: la señal atribuible al mecanismo es indistinguible del ruido. A1 rescató 9 errores de alta confianza frente a 9.780 del campeón. El cierre fue CLOSED por el árbol pre-registrado, sin segunda partición, y la reapertura V32 quedó agotada. Se marca [DOC] porque el entrenamiento consumió 1149 s de CPU y exige la matriz 279 congelada (`experiments.csv:173`; `DECISIONS.md:2823-2849`, D-082; `RUNLOG.md:2213`).

### 4.3.3 [DOC] FTRL-Proximal (V32-V35)

Cuatro ciclos pre-registrados probaron FTRL-Proximal (McMahan et al., 2013) sobre TF-IDF (Salton y Buckley, «Term-Weighting Approaches in Automatic Text Retrieval», Information Processing & Management, 1988) ajustada por fold, con la hipótesis de que el aprendizaje online proximal con L1-L2 explícito extraería otra región del espacio de texto. El primero (E3) entrenó OvR con 10 pasadas: A1 dio 0.6147759, delta -0.0011973 (1 de 5 folds), empatado bit a bit con su control barajado, y el brazo no aprendió (standalone 0.040169, por debajo del suelo trivial 0.048542). El segundo (V33) reformuló el diseño como softmax multiclase de una pasada con alpha derivado, y la puerta de convergencia G1-G6 lo cortó en el fold 0: ll_train 1.8048 frente al techo 1.6126, con colapso de frecuencias a una clase. El tercero (V34) dio la forma fuerte de la hipótesis, con alpha derivado en forma cerrada y 10 pasadas: la puerta cerró la línea por mecanismo, con G1 (ll_train de 10.09 a 19.17 frente al techo 1.7816) mientras el gemelo SGD (Bottou, «Large-Scale Machine Learning with Stochastic Gradient Descent», COMPSTAT 2010) sano descendía de 1.0622 a 0.4042; el presupuesto no era el límite, el mecanismo sí. El cuarto (V35) cubrió lambda1 > 0 con un anclaje derivado (0.23993793862904017) y la norma L2 se duplicó (razón máxima/mínima 2.1304), lo que refutó la hipótesis de acotación y cerró la familia de forma definitiva.

El cierre es por MECANISMO, no por gates de delta: la integración y el bootstrap (Efron, «Bootstrap Methods: Another Look at the Jackknife», The Annals of Statistics, 1979) no se ejecutan por pre-registro, y reabrir la familia exige un acta nueva con hipótesis mecánica nueva. La evidencia previa ya lo anunciaba: 27 celdas de alpha y 0 de 27 por debajo de H(prior) = 1.7914537. Se marca [DOC] porque exige el núcleo FTRL propio y la TF-IDF por fold, con fail-fast en el fold 0 (`experiments.csv:175, 178, 182, 190`; `DECISIONS.md:2930`, D-083; `:2852`, D-084; `:2980`, D-085; `:3072`, D-086; `RUNLOG.md:2232, 2272, 2315, 2404`).

### 4.3.4 [DOC] CORAL / cumulative-link (V23)

V23 implementó la formulación cumulative-link (Cao, Mirjalili y Raschka, «Rank consistent ordinal regression for neural networks with application to age estimation», Pattern Recognition Letters, 2020) con un único vector de pesos y cinco umbrales ordenados por construcción (véase la sección 4.1), para responder si compartir pesos entre las cinco fronteras ordinales aporta señal que los cinco binarios independientes de V20 no capturan, y si garantizar la monotonía por construcción compra métrica. El diseño incluyó dos controles: A reproduce V20 exacto (0.6097782) y B son cinco binarios lineales independientes (0.4690055), que separan el efecto de compartir pesos del de perder no linealidad.

| Modelo | macro-F1 | Delta vs V20 | Folds + | Monotonía |
|---|---|---|---|---|
| V20 (control A) | 0.6097782 | - | 3/5 | 18.927 violaciones |
| Cinco binarios lineales (control B) | 0.4690055 | -0.140773 | 0/5 | no medido |
| CORAL (peso compartido) | 0.4837017 | -0.126077 | 0/5 | 0 violaciones |

La descomposición es el resultado útil: compartir pesos y ordenar umbrales vale +0.014696 frente al control B, luego la hipótesis era correcta en su plano lineal; pero la no linealidad de LightGBM vale -0.140773, un orden de magnitud mayor. CORAL resuelve la monotonía de verdad (0 violaciones frente a 18.927) y aun así pierde 0.1261, y el mapa de errores se rehace entero (36 de 36 celdas de la matriz de confusión cambian; distancia L1 57.650; aciertos en la diagonal de 103.331 a 80.712). El cierre fue ORDINAL_SHARED_WEIGHTS = NO_SIGNAL y CORAL = CLOSED, sin submission y sin bootstrap por pre-registro: la consistencia ordinal no era el cuello de botella. Se marca [DOC] porque consume 2470 s de CPU y depende del módulo `src/ordinal_coral.py` y de la matriz 279 (`experiments.csv:129`; `DECISIONS.md:1683-1757`, D-058; `reports/v23_coral_analysis.md`; `reports/v23_coral_audit.md`).

### 4.3.5 [DOC] Modelado ordinal (V22)

V22 probó dos ideas ordinales sobre la matriz 279 de V20, sin reentrenar ningún discriminador en la segunda. La primera fue compartir representación: un único LightGBM multiclase del que se derivan las cinco CDF por suma, de modo que la monotonía queda garantizada por construcción. La segunda fue un coste de error proporcional a la distancia ordinal, que alinea la pérdida con la naturaleza ordenada de las etiquetas.

El modelo compartido dio 0.6096150, delta -0.000163 (3 de 5 folds; IC95 [-0.00078, +0.00045]; neto -13; 4.451 cambios), con 0 violaciones de monotonía frente a las 18.927 de V20: resolver el problema gratis no compró métrica. El coste ordinal fue peor: con alfa 1 el macro-F1 cayó a 0.5949198 (-0.0149; neto -3.995) y con alfa 2 a 0.5392168 (-0.0706; neto -13.838), porque los errores del modelo no están concentrados a grandes distancias y forzar su evitación degrada la frontera y la global a la vez. La decisión D-057 fue WRONG_OBJECTIVE, y con ella quedó cerrada la familia ordinal en cuatro subfamilias, junto con D-054, D-055 y D-056. Se marca [DOC] porque las dos corridas duran 33 s y 32 s, pero operan sobre la matriz 279 congelada (`experiments.csv:124, 127`; `DECISIONS.md:1633-1682`, D-057).

### 4.3.6 [DOC] Especialistas de frontera (V13-P3a y V22-B1)

Dos generaciones de especialistas binarios de frontera se probaron y cerraron. P3a entrenó un clasificador binario por par de fronteras sobre las mismas 279 variables de V12, con la hipótesis de que un modelo dedicado al par vencería a la predicción de seis clases restringida al par. El resultado fue negativo en las dos fronteras centrales: 0.4963 a 0.4944 (LogReg) y 0.4946 (MLP) en 18-24|25-34, con deltas de par de -0.0017 y -0.0029 en la variante MLP; 0.4974 a 0.4942 y 0.4945 en 25-34|35-49; y la global bajó de 0.6005 a 0.5993 con 6.944 filas modificadas. La pista se cerró: el especialista dedicado era peor que V12 restringida al par. La corrección D-048 precisó después el diagnóstico: sin la compuerta, el especialista sobre las 279 rinde 0.7136 y 0.7493 (frente a 0.6864 y 0.7218 del TF-IDF disperso), luego la señal existe y el problema era de identificabilidad, no de información.

La segunda generación (V22-B1) sustituyó solo el primer discriminador, con las otras cuatro fronteras congeladas: LinearSVC C 0.2 dio 0.5992098 (-0.0106; B1 0.6782), C 1.0 dio 0.5992058 (B1 0.6784) y un boosting especializado dio 0.6097377 con B1 0.7606, por debajo del 0.7632 de V20. La decisión fue B1_SPECIALIST = NO_SIGNAL, con razón estructural: las cinco CDF comparten features y modelo, y el primer umbral del multiclase ya está a su nivel. Se marca [DOC] porque ambos bloques exigen la matriz 279 congelada (P3a: 154 s; B1: 33 s) (`experiments.csv:86, 103, 125-126`; `DECISIONS.md:1142`, D-048; `reports/v3_especialistas_franjas.csv`).

### 4.3.7 [DOC] Canales lingüísticos (V31)

V31 construyó un banco de representaciones con modelo, folds, semilla y vectorizador fijos (LinearSVC C = 1.0; Fan et al., «LIBLINEAR: A Library for Large Linear Classification», Journal of Machine Learning Research, 2008; folds canónicos de V20) para separar "falta señal" de "falta capacidad", y probó dos vías de integración sobre V30. La hipótesis era que la edad se manifiesta en canales que el TF-IDF de palabras no expresa: URL, ortografía, composición, discurso, posición y forma de palabra. El banco (base `raw` = 0.4979941) dio la tabla siguiente.

| Banco | Representación | macro-F1 | Delta vs raw | Folds + |
|---|---|---|---|---|
| v31_02 URL | url_canal | 0.4987321 | +0.000738 | 3/5 |
| v31_02 URL | url_struct | 0.4926034 | -0.005391 | 0/5 |
| v31_02 URL | url_mask | 0.4910249 | -0.006969 | 0/5 |
| v31_02 URL | solo_url | 0.1563043 | -0.341690 | 0/5 |
| v31_03 canales | ortografía | 0.4999868 | +0.0019927 | 5/5 |
| v31_03 canales | composición | 0.5016523 | +0.0036582 | 5/5 |
| v31_03 canales | discurso | 0.5030002 | +0.0050061 | 5/5 |
| v31_03 canales | ortografía+composición+discurso | 0.5048142 | +0.0068201 | 5/5 |
| v31_03 canales | url (como canal) | 0.4998877 | +0.0018936 | 5/5 |
| v31_03 canales | posición | 0.4956907 | -0.0023034 | 0/5 |
| v31_03 canales | forma_palabra | 0.4950099 | -0.0029842 | 0/5 |
| v31_03 canales | todos | 0.4961184 | -0.0018757 | 1/5 |

Las tres familias buenas (ortografía, composición y discurso) son señal real y consistente: su combinación bate al texto solo en +0.0068201 con 5 de 5 folds, y el canal URL explícito aporta +0.0018936. Posición y forma de palabra salieron negativas y se cerraron. En la integración, el canal como cuarto miembro (N1) dio +0.0000642 sobre el control reoptimizado: la mayor parte del aparente +0.0004119 frente a V30 congelado era el efecto de reoptimizar pesos, no el canal. Como 48 features del meta-espacio (N2) dio +0.0002948 sobre el control barajado. Ambos quedan muy por debajo del gate de +0.0015. La decisión D-067 fue: URL CLOSED; posición y forma_palabra CLOSED; ortografía, composición y discurso MARGINAL (señal real sin sitio donde V30 no la exprese ya); 0 finalistas y 0 submissions. Se marca [DOC] porque el banco consumió 3802 s y la integración exige los OOF de V30 (`reports/v31_representation_benchmark.csv`; `experiments.csv:143-148`; `DECISIONS.md:2086-2152`, D-067).

### 4.3.8 [DOC] Selección supervisada de features (V25)

V25 respondió una pregunta acotada: ¿contiene V20 features ruidosas que, al eliminarse con selección supervisada dentro de cada fold, mejoren el macro-F1? La comparación usó el mismo código y los mismos folds de V20, con una sola variable: T1 conservó el top 50 % por `mutual_info_classif` (K = 139 de 279), estandarizado dentro del training fold. El control T0 reprodujo V20 exacto (0.6097782) y T1 dio 0.6091492: delta -0.000629, 1 de 5 folds positivos, 1.042 rescates contra 1.143 daños, neto -101 y 3.545 filas cambiadas.

El hallazgo por bloque explica el fracaso: sobreviven con frecuencia 1.000 las 78 probabilidades del ensamble y las 6 de NB-SVM, y nunca sobreviven 74 de las 87 estructuradas ni 30 de las 35 densas; 137 features fueron estables en los cinco folds. La selección no elimina ruido, elimina señal: lo que el mutual information ordena son las predicciones de los miembros que sostienen el modelo. La decisión fue SUPERVISED_FEATURE_SELECTION = CLOSED, sin probar otro K ni otro selector y sin bootstrap por pre-registro. Se marca [DOC] porque la corrida consume 953 s y exige la matriz 279 congelada (`experiments.csv:131`; `DECISIONS.md:1816-1882`, D-060; `reports/v25_feature_selection.md`).
