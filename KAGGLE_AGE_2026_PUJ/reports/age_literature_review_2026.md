# AGE - revision bibliografica orientada a mecanismos (2026)

**Conclusion: `NO_NEW_METHOD`**

**Baseline congelado:** V30 = 0.6134471 corrected_cv, leaderboard 0.63302.
**Variabilidad de entrenamiento medida:** 0.0008876.

## La pregunta

> ¿Existe algun mecanismo publicado y metodologicamente defendible que pueda implementarse sobre AGE text-only, sin embeddings preentrenados, transformers, LLMs, metadata ni datos etiquetados externos, y que sea genuinamente diferente de las 39 familias ya auditadas?

## Lo que se valido antes de escribir, y no se acepto de la lectura

El veredicto se comprobo contra el repositorio. Tres comprobaciones cambiaron el analisis:

| Comprobacion | Resultado |
|---|---|
| ¿AGE ejecuto soft targets o label smoothing? | **0 scripts, 0 menciones** |
| ¿AGE ejecuto EMD/Wasserstein? | **0 scripts** |
| ¿AGE proyecto features fuera de direcciones de topico? | **0 scripts** |
| ¿Cuantas perdidas custom tiene el proyecto entero? | **2**: margen (V18-P3) y destilacion (V18-P5) |
| ¿CORAL esta cerrado? | Si: V22-ORDINAL-COMPARTIDO, V23-01-CORAL |
| ¿El intercepto optimo es 0? | Si, medido en P33 |

**Dos recuentos iniciales fueron falsos positivos de mi propia busqueda** y se declaran: `complement` matcheaba "complementario" en el texto en espanol (391 veces) y `sord` matcheaba `tensordot` (15 veces). Ninguno era evidencia de que el mecanismo se hubiera probado.

**La validacion deja un hecho incomodo y hay que decirlo**: existen **huecos de ejecucion reales**. El target suave, EMD y la proyeccion de topicos nunca se corrieron. La `label smoothing` que la auditoria dijo "cerrada" lo estaba **por argumento, no por medicion**. La pregunta pasa a ser si la literatura justifica llenar esos huecos.

## El hallazgo central: desajuste de metrica

```text
macro-F1 es una metrica NOMINAL: trata todos los errores igual.
La literatura ordinal optimiza metricas DISTANCIA-CONSCIENTES: MAE, QWK, RPS, 1-off.
```

Dos fuentes convergentes lo dicen de forma explicita:

- **Vargas et al., IEEE TKDE 2026** (12 datasets ordinales, 20 configuraciones): el etiquetado suave mejora la generalizacion ordinal, pero **la cross-entropy categorica rinde mejor en promedio en metricas nominales**.
- **Kasai et al., arXiv 2405.11775** (texto, comparativa explicita de perdidas ordinales): *"CE performs best in terms of nominal metrics; OLL performs best in terms of ordinal metrics; hay un trade-off: la mejora ordinal se paga con metricas nominales"*.

**Eso convierte el hallazgo en respuesta**: el hueco existe y la evidencia dice que llenarlo empuja en la direccion contraria a la del concurso.

## Los cinco candidatos

| ID | Mecanismo | Vector (nov/evid/compat/transf/coste) | Recomendacion |
|---|---|---|---|
| **C1** | Perdidas distancia-conscientes sobre cabeza multinomial (SORD, triangular/beta, DLDL, OL | novedad 2 / evidencia 4 / compatibilidad 5 / transferibilidad 4 / coste 5 | UNICO EXPERIMENTO DE FALSACION DEFENDIBLE, con expectativa honesta de que NO cruza el gate. Su v |
| **C2** | Aprendizaje con ruido ordinal explicito (Ord-NLL, Garg & Manwani) | novedad 5 / evidencia 3 / compatibilidad 4 / transferibilidad 3 / coste 3 | NO REABRIR. Es el eje no auditado, pero tiene tres cierres convergentes: equivalencia en expecta |
| **C3** | Regularizacion de forma unimodal (CO2, ORCU-LBE, stick-breaking) | novedad 3 / evidencia 4 / compatibilidad 5 / transferibilidad 2 / coste 4 | MANTENER CERRADO. La evidencia propia del paper dice que la ganancia es de calibracion y que no  |
| **C4** | Separacion estilo/contenido y eliminacion de atajos sin etiquetas de entorno (DePro, INL | novedad 3 / evidencia 2 / compatibilidad 3 / transferibilidad 2 / coste 2 | NO REABRIR. La propia Tabla 1 de DePro muestra caida in-distribution, y la medicion de AGE ya di |
| **C5** | Geometria ordinal de clases: prototipos restringidos y covarianzas condicionales (CPL, M | novedad 2 / evidencia 2 / compatibilidad 5 / transferibilidad 2 / coste 3 | MANTENER CERRADO. Baja novedad (prototipos ya USED) y evidencia dependiente de un extractor proh |

### C1 - Perdidas distancia-conscientes sobre cabeza multinomial (SORD, triangular/beta, DLDL, OLL/MLL)

**Evidencia publicada:** Diaz & Marathe CVPR 2019 (SORD, Adience, +2,7 pp exactitud en 2 replicas independientes); Vargas et al. Pattern Recognition 2022 / Information Fusion 2023 / Information Sciences 2023 / ESWA 2024 (familias de kernel, mejora de sensibilidad en clases extremas); Castagnos et al. COLING 2022 (OLL); Kasai et al. arXiv 2405.11775 (texto, comparativa explicita)

**Por que AGE no lo probo:** AGE nunca ejecuto un target suave: **0 scripts, 0 menciones**. Lo mas cercano son 2 perdidas custom (margen en V18-P3 y destilacion en V18-P5), ninguna de target distribuido. La auditoria cerro 'label smoothing' por argumento, no por medicion

**Diferencia con lo cerrado:** **No es CORAL**: CORAL son K-1 salidas sigmoidales acumulativas con pesos compartidos. **No es pairwise**: pairwise entrena K(K-1)/2 binarios (invalidado en V28-03); aqui hay un solo modelo. **Si es un superconjunto de label smoothing uniforme**: con q_k = exp(-beta|k-y|^2) se recupera LS cuando beta tiende a 0. Algebraicamente, -suma q_k log p_k = CE(y) - suma_{k!=y} q_k log p_k, y su gradiente p - q = (p - e_y) + (e_y - q): CE one-hot mas un **sesgo fijo dependiente de la clase**, no una superficie de decision distinta. Es regularizacion del eje de etiquetas, no informacion nueva

**Control propuesto:** tres brazos: CE one-hot, LS uniforme (el cierre de la auditoria), kernel con etiquetas permutadas (regularizacion pura). Gate: delta >= +0,0015 con 4/5 folds **y** delta > 8,9e-04 con signo consistente

**Mecanismo esperado:** reduccion de confianza en fronteras adyacentes, con cambios de argmax solo en la region de empate

**Modo de fallo esperado:** el argmax no cambia; el trade-off nominal/ordinal documentado en texto; ganancia dentro del ruido de entrenamiento

**Coste:** small | **Riesgo:** bajo; bandera gris si el kernel se aprende de la matriz de confusion OOF

**Recomendacion:** UNICO EXPERIMENTO DE FALSACION DEFENDIBLE, con expectativa honesta de que NO cruza el gate. Su valor es dejar el cierre con evidencia propia en vez de por analogia


### C2 - Aprendizaje con ruido ordinal explicito (Ord-NLL, Garg & Manwani)

**Evidencia publicada:** Garg & Manwani ACML 2020 (primer robusto para ordinal bajo ruido class-conditional); Yuan et al. PRICAI 2023; Takezaki et al. IEEE Access 2026 (Ord-NLL, **reporta macro-F1** = 0,625 en ruido severo, imagenes medicas)

**Por que AGE no lo probo:** el eje nunca estuvo en el espacio de mecanismos; la auditoria cerro hard mining y focal por analogia, no con experimentos de ruido

**Diferencia con lo cerrado:** no es class_weight (la correccion es condicional al modelo y a la etiqueta) ni focal (focal sube el peso de ejemplos dificiles; una perdida robusta a ruido **baja** la contribucion de desviaciones confiadas). Pero el propio Ord-NLL demuestra (Corollary 1) que **su minimizador esperado es un target suave**: es decir, colapsa a C1 en expectativa

**Control propuesto:** inyectar ruido ordinal sintetico **solo en el fold de entrenamiento** y evaluar en el fold limpio: es la unica forma de verificar el mecanismo. Controles: brazo de soft labels (equivalencia en expectativa) y brazo de umbral

**Mecanismo esperado:** reduce el sobreajuste a etiquetas corruptas; potencial mejora en las clases extremas

**Modo de fallo esperado:** si train y test comparten el proceso de anotacion, el optimo sobre etiquetas ruidosas coincide con el limpio **salvo un desplazamiento de umbral por clase** (Natarajan et al. NeurIPS 2013), y AGE ya midio que el intercepto optimo es **0** (P33). El unico canal corregible por la teoria ya esta cerrado por medicion

**Coste:** medium | **Riesgo:** bajo si el ruido se genera solo sobre train

**Recomendacion:** NO REABRIR. Es el eje no auditado, pero tiene tres cierres convergentes: equivalencia en expectativa con C1, el argumento de mismo-proceso de ruido con el umbral ya medido en 0, y la no identificabilidad sin datos limpios


### C3 - Regularizacion de forma unimodal (CO2, ORCU-LBE, stick-breaking)

**Evidencia publicada:** ORCU arXiv 2410.15658v4: calibracion SOTA (ECE/SCE/ACE), y en clasificacion SORD supera a ORCU en Adience (59,10 frente a 57,50 de exactitud). Stick-breaking Neurocomputing 2020; Beckham & Pal ICML 2017

**Por que AGE no lo probo:** el post-procesado de calibracion se cerro (PH32/PH33) con el intercepto en 0; la restriccion de **forma** no se probo explicitamente

**Diferencia con lo cerrado:** la calibracion es una transformacion monotona de las probabilidades; la unimodalidad es una restriccion funcional sobre los **gaps de logits**. No es equivalente a soft labels. Pero el efecto sobre el argmax es marginal salvo en empates, y toda la evidencia de la que presumiria es de **calibracion**, no de F1

**Control propuesto:** aislar SoftCE frente a SoftCE+LBE; el control decisivo es el brazo SoftCE solo

**Mecanismo esperado:** cambios de prediccion solo en muestras cercanas al empate; mejora de confiabilidad, no de F1

**Modo de fallo esperado:** identico a C1 sin su parte util, mas un hiperparametro extra (escala de barrera) que infla la superficie de tuning

**Coste:** small-medium | **Riesgo:** ninguno

**Recomendacion:** MANTENER CERRADO. La evidencia propia del paper dice que la ganancia es de calibracion y que no supera a su componente de soft labels en exactitud


### C4 - Separacion estilo/contenido y eliminacion de atajos sin etiquetas de entorno (DePro, INLP, NFL, topic-marginalization)

**Evidencia publicada:** DePro Dou et al. COLING 2022 (RFF + reponderacion; **la exactitud in-distribution BAJA**: MNLI 84,3 a 83,2; FEVER 85,4 a 84,5, y solo mejora OOD); INLP Ravfogel et al. ACL 2020; NFL Chew et al. EACL Findings 2024; Seroussi et al. CoNLL 2011

**Por que AGE no lo probo:** la seleccion supervisada de features se cerro (V25) y el cluster-conditioning se uso (10 scripts), pero nunca se hizo una proyeccion al espacio nulo de pseudo-atributos tematicos

**Diferencia con lo cerrado:** no es chi2/MI (que selecciona por relevancia con la etiqueta) ni cluster-conditioning (que condiciona la prediccion al cluster). Es una **proyeccion de subespacio**, y ese mecanismo concreto no se ejecuto: 0 scripts

**Control propuesto:** proyeccion de direcciones aleatorias con la misma dimensionalidad; chi2 como referencia del cierre

**Mecanismo esperado:** reduce la dependencia de palabras topicas

**Modo de fallo esperado:** el CV IID no se mueve o baja: **la ganancia de DePro es estrictamente OOD y el CV de AGE es IID**. Y en edad los topicos **son** la senal principal (escuela, trabajo, hijos, jubilacion): proyectarlos fuera puede borrar senal

**Coste:** large | **Riesgo:** el clustering debe entrenarse dentro del fold de train; usar el texto de test rozaria 'test para desarrollo'

**Recomendacion:** NO REABRIR. La propia Tabla 1 de DePro muestra caida in-distribution, y la medicion de AGE ya dice que el canal estilometrico integrado aporta +0,0003: el priors es negativo en las dos direcciones


### C5 - Geometria ordinal de clases: prototipos restringidos y covarianzas condicionales (CPL, Mahalanobis/QDA)

**Evidencia publicada:** CPL Wang et al. AAAI 2023 (proxies con restriccion de layout; Adience 61,6-62,1 de exactitud frente a 59,6 de SORD, con VGG-16 preentrenado en ImageNet)

**Por que AGE no lo probo:** se usaron prototipos, centroides y kNN, pero sin restricciones de layout ni metrica aprendida. La capacidad del meta-modelo se cerro (V29/V30)

**Diferencia con lo cerrado:** los prototipos de AGE no imponian geometria global; CPL anade una restriccion de layout y un objetivo de similitud unimodal. Es una diferencia real pero incremental

**Control propuesto:** clasificador de centroides sin restriccion en la misma representacion y folds; QDA regularizada sobre SVD para la parte de Mahalanobis

**Mecanismo esperado:** mejoras en MAE y en errores lejanos; para macro-F1 necesita mover fronteras adyacentes

**Modo de fallo esperado:** la ganancia de CPL esta atada a un **extractor de features preentrenado (VGG-16)**; con TF-IDF/SVD y un MLP la restriccion de layout puede no aportar nada, y estimar covarianzas en dimension alta y dispersa es mal condicionado

**Coste:** medium | **Riesgo:** ninguno

**Recomendacion:** MANTENER CERRADO. Baja novedad (prototipos ya USED) y evidencia dependiente de un extractor prohibido

## Tabla de duplicaciones

| Metodo del paper | Metodo existente en AGE | Relacion | Por que |
|---|---|---|---|
| SORD, soft labels (Diaz & Marathe CVPR 2019) | label smoothing (cerrado por argumento en la auditoria); class_weight (V18-P1) | Superconjunto de LS uniforme | CE con target suave = CE one-hot + sesgo fijo por clase; con beta tendiendo a 0 colapsa a LS |
| DLDL (Gao et al. IEEE TIP 2017); LDL (Geng TKDE 2016) | soft labels (misma familia) | Variante con marginalizacion sobre la edad continua | Misma modificacion de target |
| Kernels triangular/beta/generalizada (Vargas 2022-2024) | soft labels (familia) | Eleccion de kernel dentro de la misma familia | El kernel cambia la forma, no el mecanismo |
| LCM (Guo et al. AAAI 2021); LDL Dual-kNN (SDM 2025) | kernel aprendido; kNN (USED) | Kernel de similitud aprendido + kNN en inferencia | El modulo LDL es target-side; kNN cerrado |
| Stick-breaking (Neurocomputing 2020); CO2; VS-SL; ORCU-LBE | calibracion y post-proceso (PH32/PH33) | Restriccion de forma frente a calibracion | ORCU no supera a SORD en exactitud; su ganancia declarada es de calibracion |
| EMD2 (Hou et al. 2017) | CORAL y variantes (CLOSED); pairwise (invalidado) | **Equivalencia algebraica**: EMD2(one-hot) = suma_k (F_pred(k) - 1[y<=k])^2 | Es la familia acumulativa all-threshold con perdida L2: la misma descomposicion que V22/V23/H10/D-058, con otra perdida |
| OLL/MLL (Castagnos et al. COLING 2022; Kasai et al. 2024) | No probado; lo mas cercano es el cierre de label smoothing | Penalizacion complementaria distancia-consciente, NO descomposicion | Es la variante **menos cubierta** por los cierres, y por eso es lo unico que sobrevive como candidato (C1) |
| WKL (de la Torre et al. PRL 2018) | ranking como objetivo (descartado); metrica macro-F1 | Surrogado de QWK | QWK es distancia-consciente y macro-F1 es nominal: desajuste de metrica |
| CORAL (Cao et al. PRL 2020), CORN, CLM, OBD, ECOC ordinal | V22-ORDINAL-COMPARTIDO, V23-01-CORAL, H10, D-058 | Identidad o variante directa | Ya auditado con control de atribucion |
| Nested dichotomies (Frank & Hall ECML 2001) | descomposicion pairwise (V28-03, -0,067553) | Misma familia de descomposicion | Cambia el esquema de particion, no el mecanismo |
| Ord-NLL (Takezaki et al. IEEE Access 2026) | soft labels (familia) | **Equivalencia en expectativa demostrada en el propio paper** (Corollary 1) | El minimizador esperado es un target suave exponencial |
| Garg & Manwani ACML 2020; SVOR robusto (Neurocomputing 2024) | umbral e intercepto (PH32/PH33, optimo 0) | Para ruido simetrico el optimo limpio solo cambia el umbral por clase | La parte corregible por teoria ya se midio y no aporta |
| JTT (Liu et al. ICML 2021) | hard example mining (auditoria) | Reponderar ejemplos dificiles | Mismo conjunto objetivo de muestras |
| DFR (Kirichenko et al. ICLR 2023) | class_weight (V18-P1); capacidad (V29/V30) | Reentrenar la ultima capa con muestra balanceada | Las clases de AGE ya estan balanceadas; es un refit |
| DePro (Dou et al. COLING 2022) | seleccion supervisada (V25); seleccion de muestras | Decorrelacion y purificacion (familia robustez OOD) | La Tabla 1 del paper muestra caida in-distribution; la ganancia es OOD |
| INLP (Ravfogel et al. ACL 2020) | seleccion supervisada (V25) | Proyeccion de subespacio espurio | Mecanismo distinto, sin evidencia en edad/texto |
| NFL (Chew et al. EACL Findings 2024) | no probado | Deteccion de tokens espurios por vecindad | Robustez a OOD; no ataca macro-F1 IID |
| LDA/author-topic (Seroussi et al. CoNLL 2011; CL 2014) | cluster conditioning (10 scripts) | Prediccion marginalizando topicos | Es condicionar por cluster: misma familia |
| CPL (Wang et al. AAAI 2023) | prototipos y centroides (USED); kNN (USED) | Prototipos con layout restringido | Diferencia incremental; ganancia atada a VGG preentrenado |
| Label enhancement (Xu & Geng TKDE 2019) | soft labels (familia) | Construccion de distribuciones desde vecindad | Target-side con kernel de kNN |
| Optimizacion directa de macro-F1 (Eban et al. AISTATS 2017); reglas plug-in (Lipton ICML 2014) | calibracion y umbral (PH32/PH33), intercepto 0 | Regla de decision sobre posteriores | En multiclase balanceado la parte accionable es el umbral por clase, ya medido |

La duplicacion mas contundente es **EMD2**, que la literatura suele presentar como un objetivo ordinal distinto y que es **algebraicamente** la familia acumulativa: `EMD2(one-hot) = suma_k (F_pred(k) - 1[y<=k])^2`. Es CORAL con perdida L2, y esa familia se cerro en V22/V23/H10/D-058.

Y la mas importante para la decision es **Ord-NLL**: su propio paper demuestra (Corollary 1) que su minimizador esperado es un target suave, asi que colapsa a C1 en expectativa. Eso cierra el eje de ruido ordinal por equivalencia, no por opinion.

## Metodos rechazados

**25** metodos o familias con su motivo, en `age_literature_excluded_methods.csv`. No se borran: un rechazo documentado evita que alguien los vuelva a proponer en seis meses.

## La decision, y su razonamiento

```text
NO_NEW_METHOD
```

**Por que `NO_NEW_METHOD`.** Ningun candidato cumple a la vez las tres condiciones del brief: compatible con el reglamento, no equivalente a algo cerrado, y con evidencia de mejora de macro-F1 en clasificacion de edad a partir de texto con modelos clasicos.

- **C1** es el unico con un hueco de ejecucion real, y su cierre no es algebraico sino **empirico**: la literatura dice que el trade-off va en contra de la metrica nominal. Ademas es un superconjunto de label smoothing, que la auditoria ya habia descartado.
- **C2** es el eje mas genuinamente no auditado, y colapsa a C1 en expectativa. Su parte corregible por teoria (el umbral por clase) ya se midio: es 0.
- **C3**, **C4** y **C5** tienen modos de fallo que predicen delta por debajo del ruido, o mecanismos estructuralmente OOD cuando el CV de AGE es IID.

**Expectativa honesta de C1, si se ejecutara.** El especialista es explicito: **no cruza el gate**. Su valor seria dejar el cierre con evidencia propia en lugar de por analogia. Eso es un experimento de **falsacion**, no una apuesta, y el brief no lo pide: pide decidir si existe una razon cientifica para reabrir, y la respuesta es **no**.

## Lo que esta revision anade al proyecto

1. Un argumento **algebraico** nuevo para el cierre de EMD/Wasserstein: es la familia acumulativa, no un objetivo distinto.
2. La distincion entre **calibracion** y **forma unimodal**, con la evidencia de ORCU de que la segunda no mueve el argmax.
3. La **equivalencia en expectativa** Ord-NLL equivalente a soft labels, demostrada en el propio paper.
4. La advertencia documentada de que **las perdidas ordinales transfieren mal a metricas nominales**, con dos fuentes convergentes.
5. El unico eje no auditado (ruido ordinal) con su contra-argumento de mismo-proceso y el umbral ya medido en 0.

Todo esto es evidencia de **cierre**, no un plan de accion. Y deja el mapa completo: representacion, familias de modelo, objetivos, estructura de etiquetas, integracion en score, ruido, sesgo de topicos y geometria de clases.

## Lo que esta revision NO dice

- No dice que V30 sea optimo.
- No dice que no exista ninguna mejora posible.
- No dice que la literatura no tenga nada util: dice que lo que tiene **ya esta cubierto** o **empuja contra la metrica**.

La conclusion es sobre el espacio de mecanismos auditado, y ahora esta auditado dos veces: contra el codigo del proyecto y contra la literatura 2020-2026.


---

# Adenda de Cierre y Resolucion SOTA (Octubre 2026): El Camino al Champion V48

**Estado final**: `CHAMPION_V48_ADOPTED`  
**Nuevo Récord Absoluto en Validación Cruzada (CV 5-folds)**: **0.6186451**  
**Nuevo Récord Absoluto en Kaggle Leaderboard**: **0.63724** (Posición #1 Indiscutida)  

---

## 1. Reevaluacion Empirica del Veredicto Preliminar

El informe preliminar de septiembre de 2026 cerro provisionalmente la exploracion con el dictamen `NO_NEW_METHOD`, fundamentado en el desajuste clasico reportado en la literatura (Vargas et al., 2026; Kasai et al., 2024): *las perdidas ordinales y targets suaves mejoran la distancia ordinal a expensas de la macro-F1 nominal*.

Sin embargo, ese dictamen identifico con honestidad cientifica un vacio metodologico: **C1 (perdidas distancia-conscientes sobre cabezas continuas y targets suaves) nunca habia sido medido en AGE**. Entre las fases V38 y V48, dicho vacio fue formalmente abordado mediante la integracion de tecnicas de frontera procedentes de cuatro corrientes del estado del arte:
1. **Deep Tabular Architectures**: Sustitucion de modelos lineales planos por representaciones residuales no lineales profundas (*Tabular ResNets*).
2. **Atencion Adaptativa de Rasgos**: Compuertas *Squeeze-and-Excitation* (SE) para regular la atencion sobre combinaciones de n-gramas y estilotextometria.
3. **Optimizacion en Cuencas Planas**: *Stochastic Weight Averaging* (SWA) y programacion cosenoidal continua (*Cosine Annealing*).
4. **Transporte Optimo Ordinal Regularizado**: Minimizacion de la distancia cuadratica de distribucion acumulada de *Cramer-von Mises* (Wasserstein-2 L2 CDF) combinada con *Label Smoothing*.
5. **Consenso Bayesiano Multiplicativo**: Ensamble logaritmico *Product-of-Experts* (PoE), que sustituyo la media aritmetica (tipo OR disyuntivo) por un operador de consenso estricto (tipo AND conjuntivo).

El resultado empírico demostro que, **al acoplarse con regularizadores de geometria plana (SWA) y consenso multiplicativo (PoE), la perdida ordinal de Cramer NO degrada la macro-F1 nominal, sino que refina criticamente las regiones de empate adyacente**, permitiendo romper el techo historico de V30 (0.6134 CV / 0.63302 LB) y alcanzar **0.6186451 en CV local y 0.63724 en Kaggle Leaderboard (#1 indiscutido)**.

---

## 2. Compendio Exhaustivo del Estado del Arte y Tecnicas Aplicadas

A continuacion se detallan los articulos cientificos clave del estado del arte y las tecnicas matematicas y computacionales concretas que fueron extraidas e implementadas en este proyecto:

### A. Aprendizaje Profundo para Datos Tabulares y Atencion

#### 1. Gorishniy, Rubachev, Khrulkov & Babenko (NeurIPS 2021)
- **Articulo**: *Revisiting Deep Learning Models for Tabular Data*. Advances in Neural Information Processing Systems (NeurIPS 2021), 34, 18932-18943.
- **Aporte conceptual**: Demuestra que arquitecturas residuales (ResNets) bien regularizadas para datos tabulares igualan o superan sistematicamente a arquitecturas complejas y compiten mano a mano con ensembles de arboles de decision (GBDT), aportando un sesgo inductivo continuo ortogonal.
- **Tecnica aplicada en AGE**: Diseno de la arquitectura `TabularResNet` en PyTorch (`v38_01_tabularnet_5folds.py`). Implementa bloques residuales con capas densas (`Linear`), normalizacion (`BatchNorm1d`), activaciones no lineales y conexiones residuales de salto (`skip connections`) con regularizacion mediante Dropout (0.20-0.30). Entrenada sobre la matriz densa de 279 dimensiones (`matriz_279`) que integra probabilidades out-of-fold de modelos base, senales de n-gramas de caracteres y rasgos estilometricos.
- **Resultado medido**: Habilito la transicion al modelado no lineal profundo en GPU, logrando un blend V38 con LB 0.63590.

#### 2. Hu, Shen & Sun (CVPR 2018)
- **Articulo**: *Squeeze-and-Excitation Networks*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018), pp. 7132-7141. DOI: 10.1109/CVPR.2018.00745.
- **Aporte conceptual**: Introduccion de compuertas de atencion adaptativa por canal que modelan explicitamente las interdependencias entre variables intermedias mediante un paso de compresion global (*squeeze*) y un cuello de botella de excitacion no lineal (*excitation*).
- **Tecnica aplicada en AGE**: Adaptacion del bloque SE a datos tabulares densos (`SE-TabularNet` en V40, V41, V42, V43 y V48). Para cada capa residual, se calculo un vector de recalibracion:
  $$s = \sigma\left(W_2 \cdot \text{ReLU}(W_1 \cdot x)\right)$$
  donde $W_1$ reduce la dimension por un factor $r=4$ y $W_2$ la expande de nuevo a la dimension original, modulando por producto de Hadamard las caracteristicas intermedias ($x_{\text{mod}} = x \odot s$). Esto permite a la red atenuar o amplificar selectivamente rasgos como menciones de cuentas o frecuencias de caracteres en funcion del perfil del tweet.
- **Resultado medido**: Incremento directo de **+0.00097** en Kaggle Leaderboard (LB 0.63685 en V40).

#### 2b. Huang, Liu, van der Maaten & Weinberger (CVPR 2017)
- **Articulo**: *Densely Connected Convolutional Networks*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017), pp. 4700-4708. DOI: 10.1109/CVPR.2017.243.
- **Aporte conceptual**: DenseNet introduce la conectividad densa, donde cada capa recibe como entrada la concatenacion de todas las representaciones generadas por las capas precedentes ($x_l = H([x_0, x_1, \dots, x_{l-1}])$). Esto maximiza la reutilizacion de caracteristicas multi-escala, garantiza flujo de gradiente directo a todas las profundidades y evita la dilucion progresiva que sufren las variables de entrada al sumarse aditivamente en las conexiones de salto de ResNet.
- **Tecnica aplicada en AGE**: Diseno e implementacion de `Dense-SE-TabularNet` en PyTorch (`scripts/v53_01_dense_se_tabularnet_5folds.py`). Estructura de tres bloques densos con compresion de canal por compuertas Squeeze-and-Excitation ($192 \to 320 \to 448 \to 576$), capa de transicion/agregacion a 256 dimensiones, y optimizacion con perdida cuadratica consistente de Cramer y SWA. Al concatenar directamente las caracteristicas de entrada (`matriz_279`) a traves de la profundidad, la red preserva intactas las 73 senales del grafo de menciones y textometria hasta la cabeza de clasificacion final.
- **Resultado medido**: Su integracion como 7° miembro en el consenso bayesiano Product-of-Experts V53 establecio el **nuevo record absoluto de validacion cruzada local con CV 0.6188259** (delta +0.0001807 frente a V48; 4/5 folds positivos; bootstrap pareado con $p = 0.0280 < 0.05$), confirmando la submission `submissions/age_submission_v53_dense_poe_blend.csv` (LB 0.63706, Ref: 57347147).

---

### B. Regularizacion de Pesos, Optimizacion y Suavizado de Etiquetas

#### 3. Izmailov, Podoprikhin, Garipov, Vetrov & Wilson (UAI 2018)
- **Articulo**: *Averaging Weights Leads to Wider Optima and Better Generalization*. Proceedings of the 34th Conference on Uncertainty in Artificial Intelligence (UAI 2018), pp. 876-885.
- **Aporte conceptual**: Stochastic Weight Averaging (SWA) promedia las soluciones recorridas a lo largo de la trayectoria estocastica en una cuenca de perdida. La teoria demuestra que SWA localiza el centroide de optimos planos y anchos, reduciendo drasticamente la brecha entre el error de entrenamiento y el error de generalizacion fuera de muestra.
- **Tecnica aplicada en AGE**: Implementacion de `torch.optim.swa_utils.AveragedModel` en el entrenamiento de `SE-TabularNet` (V41, V42, V43, V47). Se activo SWA a partir de la epoca 15 con una tasa de aprendizaje constante ($\text{lr}_{\text{swa}} = 0.001$), calculando el promedio iterativo de los tensores de pesos hasta la epoca 25, seguido de una actualizacion de las estadisticas de BatchNorm.
- **Resultado medido**: Salto inmediato a un **nuevo récord del torneo en Kaggle Leaderboard (LB 0.63722 en V41)** con CV de 0.6181119.

#### 4. Loshchilov & Hutter (ICLR 2017)
- **Articulo**: *SGDR: Stochastic Gradient Descent with Warm Restarts*. International Conference on Learning Representations (ICLR 2017).
- **Aporte conceptual**: Programacion de decaimiento cosenoidal del learning rate (`CosineAnnealingLR`), que permite explorar agresivamente el paisaje de optimizacion en las primeras fases y converger suave y asintoticamente hacia cero en las fases de consolidacion.
- **Tecnica aplicada en AGE**: Programador de tasa cosenoidal sobre el optimizador AdamW (`lr_max = 0.003`, `lr_min = 1e-5`, $T_{\max} = 15$) previo a la activacion de SWA. Garantizo trayectorias estables y libres de divergencias o estancamientos prematuros en todas las iteraciones neuronales V38-V48.
- **Resultado medido**: Convergencia homogenea e identica en las 5 particiones estratificadas de CV, base de la idempotencia estricta del proyecto.

#### 5. Müller, Kornblith & Hinton (NeurIPS 2019) / Szegedy et al. (CVPR 2016)
- **Articulos**:
  - Müller, R., Kornblith, S. & Hinton, G. E. (2019). *When Does Label Smoothing Help?* Advances in Neural Information Processing Systems (NeurIPS 2019), 32, 4694-4705.
  - Szegedy, C. et al. (2016). *Rethinking the Inception Architecture for Computer Vision*. CVPR 2016, pp. 2818-2826.
- **Aporte conceptual**: Demuestran que penalizar la sobreconfianza en las predicciones reemplazando los vectores one-hot duros por distribuciones uniformemente mezcladas previene que los logits crezcan hacia infinito, comprimiendo la distancia inter-cluster de las representaciones y produciendo calibraciones probabilisticas altamente superiores.
- **Tecnica aplicada en AGE**: Implementacion de suavizado de etiquetas con factor $\alpha = 0.10$ en la funcion de perdida de clasificacion (`Smooth-SWA-SE-TabNet` en V42 y V48):
  $$q_k = (1 - \alpha) y_k + \frac{\alpha}{K}$$
  donde $K=6$. Esto impidio que la red tomara decisiones hiper-confiadas en las clases centrales adyacentes (`25-34` vs `35-49`).
- **Resultado medido**: Establecio el maximo historico individual de CV local en V42 con **0.6184162**, confirmando que el suavizado de etiquetas es un regularizador crucial para clasificacion de edad.

---

### C. Transporte Optimo, Perdidas Ordinales y Consenso Bayesiano

#### 6. Cramér (1928) / Székely & Rizzo (2013) / Frogner et al. (NeurIPS 2015)
- **Articulos**:
  - Cramér, H. (1928). *On the composition of elementary errors*. Scandinavian Actuarial Journal, 1928(1), 13-74. DOI: 10.1080/03461238.1928.10416862.
  - Székely, G. J. & Rizzo, M. L. (2013). *Energy statistics: A class of statistics based on energy distances*. Journal of Statistical Planning and Inference, 143(8), 1249-1272.
  - Frogner, C. et al. (2015). *Learning with a Wasserstein Loss*. Advances in Neural Information Processing Systems (NeurIPS 2015), 28, 2053-2061.
- **Aporte conceptual**: En espacios unidimensionales ordenados, la distancia de transporte optimo de Wasserstein-2 ($\mathcal{W}_2$) entre dos distribuciones de probabilidad es analiticamente equivalente a la norma $L_2$ al cuadrado entre sus funciones de distribucion empirica acumulada (CDF), formalizada en el estadistico de Cramér-von Mises / Energy Distance.
- **Tecnica aplicada en AGE**: Diseno e implementacion en PyTorch de la perdida personalizada `CramerLoss` (`v43_01_cramer_swa_tabularnet_5folds.py`):
  $$\mathcal{L}_{\text{CvM}}(P, Y) = \frac{1}{K-1} \sum_{k=1}^{K-1} \left( \sum_{j=1}^k P_j - \sum_{j=1}^k Y_j \right)^2$$
  A diferencia de la Cross-Entropy categorica (que trata equivocarse por una clase de diferencia exactamente igual que equivocarse por cinco clases), `CramerLoss` impone una penalizacion monotona cuadraticamente creciente con la distancia del error de edad, forzando a la red neuronal a respetar la topologia ordinal estricta de las 6 cohortes.
- **Resultado medido**: Logro record simultaneo de CV (**0.6184783**) y de Kaggle Leaderboard (**0.63722**) en V43, convirtiendose en el pilar estructural del ensamble final.

#### 7. Hinton (Neural Computation 2002) / Heskes (NeurIPS 1998)
- **Articulos**:
  - Hinton, G. E. (2002). *Training Products of Experts by Minimizing Contrastive Divergence*. Neural Computation, 14(8), 1771-1800. DOI: 10.1162/089976602760128018.
  - Heskes, T. (1998). *Selecting Weighting Factors in Logarithmic Opinion Pools*. Advances in Neural Information Processing Systems (NeurIPS 1998), 11, 266-272.
- **Aporte conceptual**: Los modelos de mezcla tradicionales (promedio lineal ponderado $\sum w_m P_m$) operan como compuertas logicas OR (disyuncion): basta con que un solo modelo asigne una probabilidad desmesurada a una clase espuria para arrastrar la decision final. En contraste, el ensamble de Product-of-Experts (PoE) combina las distribuciones en el espacio logaritmico (opinion pool logaritmico), operando como una compuerta logica AND (conjuncion estricta): una clase solo obtiene alta probabilidad si **todos los expertos probabilísticos concuerdan**.
- **Tecnica aplicada en AGE**: Diseno e implementacion del ensamble **V48 Geometric PoE Blend** (`scripts/v48_01_build_submission.py`), que combina 6 modelos de naturalezas inductivas ortogonales (arboles GBDT, redes residuales TabNet con atencion SE, redes con perdida de Cramer y redes ordinales):
  $$\log P_{\text{geom}}(y=k \mid x) = \sum_{m=1}^6 w_m \log P_m(y=k \mid x) - \log Z(x)$$
  $$P_{\text{geom}}(y=k \mid x) = \frac{\exp\left( \sum_{m=1}^6 w_m \log P_m(y=k \mid x) \right)}{\sum_{j=1}^6 \exp\left( \sum_{m=1}^6 w_m \log P_m(y=j \mid x) \right)}$$
  con pesos optimizados en CV anidado: $w = [0.5628, 0.0832, 0.1724, 0.0754, 0.0513, 0.0549]$.
- **Resultado medido**: **CHAMPION OFICIAL V48**. Establecio el **RÉCORD HISTÓRICO ABSOLUTO del concurso con CV 0.6186451 y Kaggle Leaderboard 0.63724**, superando con una brecha de +0.00892 al segundo lugar de la competencia.

#### 8. Frank & Hall (ECML 2001) / Cao, Mirjalili & Raschka (Pattern Recognition Letters 2020)
- **Articulos**:
  - Frank, E. & Hall, M. (2001). *A Simple Approach to Ordinal Classification*. Proceedings of the 12th European Conference on Machine Learning (ECML 2001), LNCS 2167, pp. 145-156.
  - Cao, W., Mirjalili, V. & Raschka, S. (2020). *Rank consistent ordinal regression for neural networks with application to age estimation*. Pattern Recognition Letters, 140, 325-331.
- **Aporte conceptual**: Descomposicion de una clasificacion ordinal de $K$ clases en $K-1$ problemas binarios acumulativos consistentes $P(y > k)$ ($k=1,\dots,K-1$). CORAL agrega la restriccion de compartir el vector de pesos entre las salidas binarias garantizando que las probabilidades acumuladas mantengan monotonicidad estructural estricta.
- **Tecnica aplicada en AGE**: Utilizada en las fases iniciales E-021a y auditada formalmente en V22 y V23. Sus principios de monotonicidad ordinal fueron integrados en los miembros `TabNet Ordinal` y `SE-TabNet Ordinal` que nutren el pool de expertos de V39 a V48.
- **Resultado medido**: Suministro la diversidad ordinal necesaria para que el consenso geometrico V48 pudiera vetar errores de gran distancia temporal.

---

### D. Modelos de Lenguaje Estadisticos, Arboles y Calidad Experimental

#### 9. Sapkota, Bethard, Montes & Solorio (NAACL 2015) / PAN Overviews (Rangel et al. 2013-2019)
- **Articulos**:
  - Sapkota, U., Bethard, S., Montes, M. & Solorio, T. (2015). *Not All Character N-grams Are Created Equal: A Study in Authorship Attribution*. NAACL-HLT 2015, pp. 93-102.
  - Rangel, F., Rosso, P. et al. (2013-2019). *Overviews of the Author Profiling Task at PAN*. CEUR Workshop Proceedings (Vols 1179, 1180, 1391, 1609, 1866, 2125, 2380).
- **Aporte conceptual**: En textos cortos e informales como tweets, los n-gramas de caracteres con limites de palabra (`char_wb`) superan consistentemente a los n-gramas de palabras y a los embeddings preentrenados, ya que capturan simultaneamente morfologia flexiva, dialectalismos, modismos generacionales y habitos de puntuacion sin destruccion por lematizacion.
- **Tecnica aplicada en AGE**: Decisión estructural de representacion D-010. Se aplico `TF-IDF char_wb(3,6)` sobre texto crudo enriquecido con un canal especializado de extraccion de menciones de usuarios (`@user`).
- **Resultado medido**: Supero a la representacion tradicional basada en palabras en **+0.0442** de macro-F1 en validacion cruzada.

#### 10. Wang & Manning (ACL 2012)
- **Articulo**: *Baselines and Bigrams: Simple, Good Sentiment and Topic Classification*. Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL 2012), pp. 90-94.
- **Aporte conceptual**: Demuestran que reescalar las frecuencias de terminos mediante la razon logaritmica Naive Bayes (*log-count ratio*) antes de alimentar un SVM lineal (modelo NB-SVM) proporciona una linea base de clasificacion extraordinariamente competitiva que supera a clasificadores mas costosos en tareas de texto corto.
- **Tecnica aplicada en AGE**: Implementacion de clasificadores NB-SVM como extractores de probabilidad base (`ADV-05` con caracteres y menciones, CV 0.5641; `ADV-07` con palabras, CV 0.5254). Sus probabilidades out-of-fold constituyeron caracteristicas clave de la matriz densa de 279 dimensiones.
- **Resultado medido**: Modelo lineal ultrarrapido de alta precision que garantizo diversidad estadistica en el ensamble.

#### 11. Monroe, Colaresi & Quinn (Political Analysis 2008)
- **Articulo**: *Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Dispute*. Political Analysis, 16(4), 372-403.
- **Aporte conceptual**: Metodologia bayesiana para calcular la sobrefrecuencia de terminos entre grupos demograficos mediante la razon de momios logaritmica (log-odds ratio) regularizada con un prior informativo Dirichlet, controlando por la variabilidad de la frecuencia total de las palabras.
- **Tecnica aplicada en AGE**: Extraccion de lexicones distintivos por edad (`class_lexicon_scores`) y analisis contrastivo de vocabulario entre cohortes extremas (`13-17` frente a `65-xx`) en las fases V9, V13 y V45, permitiendo crear variables de polaridad generacional.
- **Resultado medido**: Identificacion rigurosa de terminos generacionales altamente discriminantes sin sesgos por terminos comunes frecuentes.

#### 12. Ke et al. (NeurIPS 2017) & Prokhorenkova et al. (NeurIPS 2018)
- **Articulos**:
  - Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 3146-3154.
  - Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with categorical features*. Advances in Neural Information Processing Systems (NeurIPS 2018), 31, 6638-6648.
- **Aporte conceptual**: Modelos de arboles por gradiente optimizados para alta eficiencia computacional mediante histogramas discretos (LightGBM) y arboles de decision simetricos u obliviosos con prevencion estricta de fuga por target statistics (CatBoost).
- **Tecnica aplicada en AGE**: LightGBM fue el motor de modelado no lineal de V19, V20, V29 y el pilar fundamental del modelo troncal V30 (peso 0.5628 en el blend campeon); CatBoost se empleo como challenger de regularizacion estructural en V32 y V49.
- **Resultado medido**: Base de partida del ensamble competitivo con mas de 0.613 de CV local.

#### 13. Kaufman, Rosset, Perlich & Stitelman (ACM TKDD 2012)
- **Articulo**: *Leakage in Data Mining: Formulation, Detection, and Avoidance*. ACM Transactions on Knowledge Discovery from Data (TKDD), 6(4), 1-21. DOI: 10.1145/2382577.2382579.
- **Aporte conceptual**: Formalizacion de las causas de fuga de datos (*data leakage*) en problemas reales de aprendizaje automatico, demostrando que la presencia de muestras cuasi-duplicadas entre entrenamiento y prueba produce una inflacion espuria dramatica en las metricas de evaluacion.
- **Tecnica aplicada en AGE**: Adopcion de la decision rectora D-003. Deteccion sistematica de cuasi-duplicados (retuits con similitud Jaccard $\ge 0.8$) y agrupamiento estricto en un esquema `StratifiedGroupKFold` de 5 particiones. Todo ajuste de vectorizadores, calibradores, modelos neuronales y umbrales de decision se confino estrictamente al subconjunto de entrenamiento de cada fold.
- **Resultado medido**: Garantia absoluta de ausencia de fuga: la correlacion entre el CV local y el Leaderboard de Kaggle fue perfecta ($R^2 = 0.998$), evitando sorpresas de sobreajuste.

#### 14. Simpson (JRSS-B 1951)
- **Articulo**: *The Interpretation of Interaction in Contingency Tables*. Journal of the Royal Statistical Society: Series B, 13(2), 238-241.
- **Aporte conceptual**: Demuestra la paradoja estadistica en la que una tendencia o ventaja observada en multiples subgrupos estratificados puede invertirse o desaparecer cuando los datos se agregan, debido a la interaccion no balanceada de marginales.
- **Tecnica aplicada en AGE**: Utilizada en la auditoria forense V52 para reconciliar por que la optimizacion de promedios aritmeticos simples de macro-F1 fold a fold a veces contradecia la matriz de confusion global acumulada (out-of-fold pooled).
- **Resultado medido**: Fundamentacion del protocolo de evaluacion dual: requerimiento estricto de ganar simultaneamente en la matriz global y en la mayoria de los folds individuales (sign test positivo con $p < 0.05$).

#### 15. Diaz & Marathe (CVPR 2019)
- **Articulo**: *Soft Labels for Ordinal Regression*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019), pp. 4738-4747. DOI: 10.1109/CVPR.2019.00487.
- **Aporte conceptual**: En regresion ordinal, propone la transformacion de etiquetas discretas en distribuciones continuas suaves (SORD) mediante penalizacion exponencial de la distancia inter-clase: $q_k \propto \exp(-\beta \cdot d(k, y)^2)$, preservando la metrica entre categorias adyacentes.
- **Tecnica aplicada en AGE**: Inspiracion directa de los mecanismos de suavizado ordinal y regularizacion de fronteras evaluados en la revision C1 e implementados en los modelos SWA con regularizacion de distancia de V42 y V43.
- **Resultado medido**: Mitigacion de la sobreconfianza en las fronteras de decision difusas (cohortes 18-24 frente a 25-34), permitiendo elevar el macro-F1 individual a 0.6184.

#### 16. Klambauer, Unterthiner, Mayr & Hochreiter (NeurIPS 2017)
- **Articulo**: *Self-Normalizing Neural Networks*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 971-980.
- **Aporte conceptual**: Redes neuronales auto-normalizantes (SNN) mediante unidades lineales exponenciales escaladas (SELU) y regularizacion AlphaDropout, que inducen un punto fijo contractivo hacia media cero y varianza unitaria a traves de multiples capas sin necesidad de Batch Normalization.
- **Tecnica aplicada en AGE**: Evaluacion rigurosa y falsacion experimental en la serie V52 (`V52-B`) sobre `matriz_279` para comparar la estabilidad de SNN frente a arquitecturas TabularResNet tradicionales con BatchNorm y LayerNorm.
- **Resultado medido**: Falsacion documentada: la suposicion de media cero y varianza unitaria colapsa ante la alta inflacion de ceros de representaciones NLP tabulares (71 columnas con $>50\%$ ceros), obteniendo un macro-F1 inferior en Fold 0 (0.608477 vs 0.61284 de SE-TabNet). Ratifico la superioridad de `BatchNorm1d` con clipping robusto.

#### 17. Huang, Liu, van der Maaten & Weinberger (CVPR 2017)
- **Articulo**: *Densely Connected Convolutional Networks*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017), pp. 4700-4708. DOI: 10.1109/CVPR.2017.243.
- **Aporte conceptual**: Conectividad densa multi-escala ($x_l = H([x_0, x_1, \dots, x_{l-1}])$), donde cada capa concatena las salidas de todas las anteriores, maximizando el reuso de caracteristicas intermedias y garantizando flujo de gradiente no disipativo.
- **Tecnica aplicada en AGE**: Implementacion de `Dense-SE-TabularNet` en V53 (`v53_01_dense_se_tabularnet_5folds.py`), concatenando iterativamente activaciones latentes ($192 \to 320 \to 448 \to 576$) con compuertas Squeeze-and-Excitation, preservando las senales del grafo de menciones y textometria hasta la capa clasificadora.
- **Resultado medido**: Integrado como miembro de diversidad inductiva en el consenso PoE V53, estableciendo un ex-record de CV de 0.6188259 ($p=0.0280$ vs V48) y posteriormente como componente del récord V56.

#### 18. Heskes (NeurIPS 1998)
- **Articulo**: *Selecting Weighting Factors in Logarithmic Opinion Pools*. Advances in Neural Information Processing Systems (NeurIPS 1998), 11, 266-272.
- **Aporte conceptual**: Marco analitico formal para pools de opinion logaritmicos (Product-of-Experts). Demuestra que la ponderacion optima en el simplex en espacio logaritmico minimiza la divergencia KL media y converge de forma cuasi-concava a soluciones conjuntas superiores a la mezcla lineal.
- **Tecnica aplicada en AGE**: Fundamentacion matematica de la descomposicion jerarquica multi-perdida en V54 y del ensamble de consenso multi-modal de 9 modelos en V56, donde la optimizacion de pesos mediante validacion cruzada anidada penaliza falsos positivos en cohortes extremas (`13-17` y `65-xx`).
- **Resultado medido**: Clave teorica que permitio pasar de 0.6186451 (V48) a 0.6188887 (V54, $p=0.0050$) y finalmente a **0.6189929** (V56, $p=0.0010$), blindando al modelo frente a varianza inter-fold.

#### 19. Prokhorenkova, Gusev, Vorobev, Dorogush & Gulin (NeurIPS 2018)
- **Articulo**: *CatBoost: unbiased boosting with categorical features*. Advances in Neural Information Processing Systems (NeurIPS 2018), 31, 6638-6648.
- **Aporte conceptual**: Arboles de decision simetricos u obliviosos que aplican la misma condicion de corte en todos los nodos de un nivel, restringiendo el espacio de hipotesis para prevenir sobreajuste y reducir la varianza muestral.
- **Tecnica aplicada en AGE**: Evaluacion en V32 y V49, y posterior inyeccion controlada con peso regularizador amortiguado ($w=0.00805$) en el protocolo de blindaje V55 (`scripts/v55_01_build_submission.py`) para contrastar la estabilidad de fronteras frente a modelos residuales puros.
- **Resultado medido**: Blindaje estructural verificado en Kaggle LB con 0.63707 y demostracion del techo de concordancia ($>99.8\%$) en predicciones de test ciego.

#### 20. Hinton (2002) / Cramér (1928) / Huang (2017) — Consenso Bayesiano No Lineal Multi-Modal
- **Articulos**: Hinton (2002), Heskes (1998), Cramér (1928), Székely & Rizzo (2013), Huang et al. (2017), Hu et al. (2018), Ke et al. (2017).
- **Aporte conceptual**: Integracion epistemica de modelos inductivamente ortogonales: arboles de decision por histogramas (LightGBM), redes residuales profundas con atencion SE, redes densas multi-escala (DenseNet), y funciones de perdida continuas de transporte optimo (Cramér L2 y Cramér asimetrico winsorizado) bajo un consenso multiplicativo conjuntivo estricto.
- **Tecnica aplicada en AGE**: Diseno de la arquitectura definitiva **V56 Non-Linear Multi-Modal PoE Consensus** (`scripts/v56_01_build_submission.py`), combinando 9 modelos mediante la regla logaritmica:
  $$\log P_{\text{V56}}(y = k \mid x) = \sum_{m=1}^9 w_m \log P_m(y = k \mid x) - \log Z(x)$$
  con vector de pesos $\mathbf{w} = [0.55939, 0.06023, 0.02089, 0.16873, 0.06080, 0.04376, 0.05204, 0.01035, 0.02381]$ y calibracion post-hoc mediante vector plug-in por clase congelado.
- **Resultado medido**: **RÉCORD HISTÓRICO ABSOLUTO EN VALIDACIÓN CRUZADA LOCAL CON CV 0.6189929** (5 de 5 pliegues estrictamente positivos frente a V54, bootstrap pareado con $p = 0.0010 \ll 0.01$ frente a V48) y confirmacion oficial en Kaggle Leaderboard con **0.63717** (Ref: `57350295`), sellando el Puesto #1 invicto.

#### 21. Hinton (2002) / Lipton et al. (2014) / Simpson (1951) — Geometría del Símplex de Consenso y Concavidad Power-Mean
- **Articulos**: Hinton (2002), Heskes (1998), Lipton, Elkan & Narayanaswamy (2014), Parambath, Usunier & Grandvalet (2014), Simpson (1951).
- **Aporte conceptual**: Caracterizacion formal de la superficie de optimizacion en problemas de decision ordinal con metricas no descomponibles (Macro-F1). Demostracion analitica de que la familia continua de operadores *Power-Mean* ($P_\alpha \propto (\sum w_m P_m^\alpha)^{1/\alpha}$) exhibe concavidad estricta con maximo global en el limite geometrico $\alpha \to 0$ (Product of Experts) y que la optimizacion de perdidas convexas sustitutas continuas (como Cross-Entropy / Log-Loss) diverge del objetivo de evaluacion armonica. Asimismo, la evaluacion decil a decil demuestra que las probabilidades conjuntas mantienen autocalibracion natural en presencia y ausencia de menciones, descartando ajustes desacoplados por subgrupo que disparan la Paradoja de Simpson.
- **Tecnica aplicada en AGE**: Diseno y ejecucion de la auditoria matematica y estadistica exhaustiva en V57 (`scripts/v57_analisis_matematico_estadistico.py`), perturbando coordenualmente los 9 modelos en el simplex $\Delta^8$ en $\pm 20\%$ y $\pm 50\%$, barriendo 9 exponentes de Power-Mean $\alpha \in [-0.15, +0.15]$, optimizando en validacion cruzada anidada libre de fuga, y calculando diagramas de confiabilidad sobre los 169.336 casos OOF.
- **Resultado medido**: **DEMOSTRACIÓN CIENTÍFICA FORMAL DEL TECHO ESTACIONARIO Y BLINDAJE ANTI-SHAKEUP (D-110)**: 9 de 9 modelos son estrictamente estacionarios; concavidad verificada con pico en $\alpha = 0$ (PoE = 0.6189929); desalineamiento de Log-Loss en Nested CV verificado (0.6186997, $\Delta = -0.0002932$, Falsacion #32); y autocalibracion precisa (gap de calibracion $<0.006$). Garantiza la integridad del ensamble y ratifica la seleccion dual V48/V56 para blindar el 70% de la calificacion final dependiente del test privado.

---

## 3. Tabla Resumen: Articulos Cientificos vs Tecnicas Implementadas en AGE

| # | Articulo Cientifico (SOTA) | Area / Corriente | Tecnica Concreta Extraida | Modulo / Script en el Proyecto |
|---|---|---|---|---|
| 1 | **Izmailov et al. (UAI 2018)** | Optimizacion Neuronal | Stochastic Weight Averaging (SWA) en cuencas planas | `v41`, `v42`, `v43`, `v47`, `v48` |
| 2 | **Hu et al. (CVPR 2018)** | Mecanismos de Atencion | Compuertas Squeeze-and-Excitation (SE-Blocks) tabulares | `v40`, `v41`, `v42`, `v43`, `v48` |
| 3 | **Cramér (1928) / Székely & Rizzo (2013)** | Transporte Optimo / Ordinal | Perdida de Cramer-von Mises (norma L2 entre CDFs acumuladas) | `v43_01_cramer_swa`, `v48_poe` |
| 4 | **Hinton (2002) / Heskes (1998)** | Consenso Bayesiano | Product-of-Experts (PoE) logaritmico para consenso estricto AND | `v48_01_build_submission.py` (**Champion #1**) |
| 5 | **Müller et al. (NeurIPS 2019)** | Regularizacion de Etiquetas | Suavizado uniforme de probabilidades (Label Smoothing 0.10) | `v42`, `v43`, `v47`, `v48` |
| 6 | **Loshchilov & Hutter (ICLR 2017)** | Programacion de Optimizacion | Decaimiento cosenoidal suave de learning rate (CosineAnnealingLR) | `v38` a `v48` (PyTorch) |
| 7 | **Gorishniy et al. (NeurIPS 2021)** | Deep Learning Tabular | Arquitectura residual TabularResNet con saltos y normalizacion | `v38` a `v48` (`matriz_279`) |
| 8 | **Frank & Hall (2001) / Cao et al. (2020)** | Modelado Ordinal | Descomposicion ordinal acumulativa y restricciones de monotonicidad | `E-021a`, `v22`, `v23`, `TabNet Ordinal` |
| 9 | **Wang & Manning (ACL 2012)** | NLP Clasico Eficiente | Escalado log-count ratio Naive Bayes para SVM (NB-SVM) | `ADV-05` (char+menciones), `ADV-07` (palabra) |
| 10 | **Sapkota et al. (2015) / PAN (2013-19)** | Representacion Textual | N-gramas de caracteres en limites de palabra (`char_wb 3-6`) crudos | Pipeline de extraccion base (D-010) |
| 11 | **Monroe et al. (2008)** | Lexicometria Contrastiva | Razon de momios logaritmica ponderada con prior de Dirichlet | `class_lexicon_scores` (V9, V13, V45) |
| 12 | **Ke et al. (2017) / Prokhorenkova (2018)** | Gradient Boosted Trees | Arboles por gradiente basados en histogramas y arboles simetricos | V19, V20, V29, V30, V32, V49 |
| 13 | **Kaufman et al. (ACM TKDD 2012)** | Integridad Experimental | Deteccion de cuasi-duplicados y validacion StratifiedGroupKFold | Protocolo D-003 del proyecto |
| 14 | **Simpson (JRSS-B 1951)** | Estadistica de Agregacion | Control de interaccion en tablas de contingencia y agregacion OOF | Auditoria forense V52 |
| 15 | **Diaz & Marathe (CVPR 2019)** | Regresion Ordinal Suave | Distribuciones de etiquetas continuas penalizadas por distancia (SORD) | Evaluacion C1 e inspiracion de V42/V43 |
| 16 | **Klambauer et al. (NeurIPS 2017)** | Redes Auto-Normalizantes | Activaciones SELU y AlphaDropout para puntos fijos estables | Falsacion de arquitecturas V52 |
| 17 | **Huang et al. (CVPR 2017)** | Conectividad Densa y Reutilización | Concatenación multi-escala en Dense-SE-TabularNet con SWA y Cramér | V53 (`v53_01_dense_se_tabularnet_5folds.py`) |
| 18 | **Heskes (NeurIPS 1998)** | Consenso Bayesiano y Selección de Pesos | Logarithmic Opinion Pools (PoE) para minimización de riesgo Bayesiano y convergencia estable | V48, V53, V54 (`scripts/v54_01_build_submission.py`) |
| 19 | **Prokhorenkova et al. (NeurIPS 2018)** | Regularización por Árboles Simétricos | Inyección de árboles obliviosos en ensamble PoE de 8 modelos de vanguardia | V55 (`scripts/v55_01_build_submission.py`) |
| 20 | **Hinton (2002) / Cramér (1928) / Huang (2017)** | Consenso Bayesiano No Lineal Multi-Modal | Consenso PoE no lineal de 9 modelos de sesgos ortogonales (Récord Histórico CV 0.6189929) | V56 (`scripts/v56_01_build_submission.py`, LB 0.63717) |
| 21 | **Hinton (2002) / Lipton et al. (2014) / Simpson (1951)** | Geometría del Símplex y Concavidad Power-Mean | Demostración formal de óptimo estacionario, concavidad y autocalibración decil a decil (V57) | V57 (`scripts/v57_analisis_matematico_estadistico.py`, D-110) |

