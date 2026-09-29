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
