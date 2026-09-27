# V26-01: dictamen de priorizacion y veredicto sobre el techo

Proyecto `kaggle_toxic_2026_puj`, rama `v25`, 2026-09-27. Referencia **V03 = 0.78306 LB**,
`corrected_cv` canonico 0.7772114525. Gastos de esta fase: **0 experiments, 0 submissions**.

Correccion previa aceptada: la columna `diferencia` de `submissions/leaderboard_log.csv` es
local menos kaggle. **0 de 22 submissions con score superan a V03.** El segundo mejor es
`v08_meta_v2` con 0.781630, margen -0.00143.

## 0. Resumen del dictamen

1. El techo `dOOF = +0.03756` es, en su mayor parte, un artefacto, y **no** un objetivo de
   modelado. Su intervalo de confianza al 95 % es **[+0.0216, +0.1463]**: un factor 6,8 de
   amplitud. Ver seccion 2.
2. El techo es inalcanzable, pero **no** porque V03 sea infranqueable, sino porque la
   superficie del leaderboard esta plana dentro de 1,1 SE y V03 es el **argmax de 9
   sorteos**. Ver seccion 3. Esto confirma la hipotesis del lider de equipolvante y tiene una
   consecuencia contraintuitiva: si V03 fue afortunado, el techo **sube**, no baja.
3. Se seleccionan **4 familias**, con un orden de apertura y cierre. Dos se ejecutan, una es
   una prueba predictiva de 20 minutos, y una queda bloqueada por auditoria de reglas.
4. **Ningun gate razonable autoriza una submission.** Se demuestra en la seccion 5. El
   proyecto cierra en el eje competitivo, y se cierra por potencia del instrumento, no por
   acumulacion de negativos.

## 1. Lo que el techo de +0.03756 es y lo que no es

El numero viene de `dLB = -0.00920 + 0.2448 * dOOF` (`reports/v24_01_transferencia_complejidad.md`,
seccion 4), ajustado con Pearson 0.4876, p 0.0342, n 19.

**Precision del numero.** Con r = 0.4876 y n = 19 el error estandar de la pendiente es
0.0928, luego el IC95 de la pendiente es **[0.063, 0.427]**. El `dOOF` de empate es
`0.00920 / pendiente`, y por tanto:

| extremo | pendiente | dOOF de empate | mejor dOOF historico | fraccion |
|---|---:|---:|---:|---:|
| optimista | 0.427 | +0.0216 | +0.0132 | 61 % |
| puntual | 0.2448 | **+0.0376** | +0.0132 | 35 % |
| pesimista | 0.063 | +0.1463 | +0.0132 | 9 % |

Tres consecuencias.

1. **El +0.03756 no es una constante, es el centro de una banda que spans un factor 6,8.** Un
   gate fijado en el punto medio de una banda de esa amplitud no es un gate, es un Preferred
   Citation. Ademas +0.03756 esta **1,4 veces fuera del rango observado** de `dOOF` (maximo
   observado +0.0267 en V05, y +0.0132 con la recalculacion canonica): es una
   extrapolacion, no una medicion.
2. **El intercepto negativo esta forzado, como ya dijo el dictamen anterior.** Con 19
   perdedores y un ganador, cualquier recta ajustada por minimos cuadrados **debe** cortar el
   eje por debajo del maximo observado. El intercepto -0.00920 no es un parametro del
   problema: es una propiedad de la muestra. Reformular "V03 fue afortunado" como "hace falta
   dOOF = +0.0376" es la misma frase dos veces, y por eso el techo **se mueve con V03**, no
   con el dataset.
3. **El extremo optimista es el unico que importa, y es incompatible con la unica medicion
   directa.** El IC bajo exige una tasa de transferencia de 0,427, o sea que dOOF se
   convierta en LB casi 1:1. Pero V23 es el unico punto emparejado con un `dLB` medido, y
   midio lo contrario: `dOOF = +0.013199` produjo una calidad realizada de **+0.00158** sobre
   la media del campo, o sea una tasa efectiva de **0,12**, la mitad de la ajustada. El
   extremo optimista del IC es el extremo que la medicion directa ya contradijo.

**Consecuencia practica.** El techo Util es un intervalo, no un punto. Y en las tres
versiones de ese intervalo el mejor `dOOF` del proyecto queda entre el **9 % y el 61 %** de lo
necesario. En ninguna version el techo esta al alcance de lo demostrado.

## 2. La pregunta del lider: es alcanzable, o es un artefacto del ajuste

**Es un artefacto, y el diagnostico del proyecto era correcto.** Pero el juicio de ese diagnostico
tiene una consecuencia que el dictamen anterior no derivo de forma explicita, y que es lo que decide esta
fase.

Cuando V03 "fue afortunado", el liston **no baja**: V03 se queda en 0.78306, que es un hecho
medido y exacto sobre las 1154 filas del test, no una estimacion. Un candidato nuevo tiene
que superar ese hecho, no la calidad verdadera que se le atribuye a V03. O sea, la
interpretacion "V03 fue afortunado" **endurece** el objetivo, y por eso el techo de +0.0376
es alcanzable en un sentido y en el otro no: es alcanzable por un modelo que sea
verdaderamente mucho mejor, y la sorpresa es que **nadie ha demostrado ni de lejos esa
magnitud**.

## 3. La medicion que decide: la distribucion del campo, y por que V03 no tiene skill

Esto no estaba medido en el repositorio. Se calcula sobre `submissions/leaderboard_log.csv`,
descartando la plantilla (0.51494) y **descartando tambien las sondas de V16**, porque las
cuatro (C=1,5 / C=2,5 / min_df=1 / ajustes de tasa) son perturbaciones disenadas para ser
inferiores a V03, no sorteos. Incluirlas inflaria la varianza por diseno.

**Los 9 candidatos estructuralmente distintos** (v04, v05, v08_meta_v2, v09_candidate_a,
v10_candidate_b, v11_candidate_c, v12_p7, v20_01, v23_01):

| magnitud | valor |
|---|---:|
| n | 9 |
| media | 0.777801 |
| desviacion tipica | 0.003377 |
| minimo (v12_p7) | 0.771220 |
| maximo (v08_meta_v2) | 0.781630 |
| **V03** | **0.783060** |

**El contraste decisivo.** La z de V03 sobre el campo es `(0.78306 - 0.777801) / 0.003377` =
**+1,557**. El maximo esperado de **9 sorteos de una normal** es
`Phi^-1((9-0,375)/(9+0,25))` = **+1,494**. Y `P(max <= 1,557 | nueve sorteos de ruido puro)`
= **0,575**.

**V03 esta exactamente donde deberia estar si no tuviera ninguna habilidad y fuera
simplemente el mejor de nueve dados la vuelta.** No es un outlier incluso bajo la nula. La
nula "todos los candidatos tienen la misma calidad verdadera y el reparto es ruido de test"
no solo no se rechaza: es la explicacion que mejor encaja. Y el margen de V08 (-0,00143) no
es un casi-acierto, es un sorteo mas.

Sobre las 21 submissions de modelo (incluyendo las sondas de V16) la media es 0.775625 y la
sd 0.005740, luego **la probabilidad de que un candidato nuevosortea de la misma poblacion
supere a V03 es 1 - Phi(1,296) = 0,098**, o sea un 10 %. Para tener p >= 0,50 hace falta una
calidad realizada de **+0,0038** sobre la media del campo.

**El contraste con la unica medicion de campo abierto.** V23 es la candidata con mejor
evidencia interna de la historia y su calidad realizada fue **+0,00158** sobre la media del
campo. La mejora que hay que gainar para superar a V03 es **+0,00526**. O sea
**3,3 veces la mejor mejora realizada del proyecto**, y 1,56 sd de un campo cuya propia
dispersion es en su mayor parte ruido.

**Tres rutas independientes, un mismo factor.**

| ruta | estimativa del `dOOF` necesario | mejor `dOOF` historico | fraccion |
|---|---:|---:|---:|
| regresion de V24, punto | +0.0376 | +0.0132 | 35 % |
| regresion de V24, IC95 optimista | +0.0216 | +0.0132 | 61 % |
| tasa de intercambio medida en V23 | +0.0440 | +0.0132 | 30 % |

El factor que falta es ~3, y **no se ha cerrado ni una vez en 22 intentos**. Este es el
argumento que decide, y es un argumento de potencia, no dehartismo.

## 4. Familias seleccionadas

Criterio de seleccion: no probadas en el repositorio (verificado por grep sobre 156 scripts
`.py` y 166 reports `.md`), sesgo inductivo explicito y no lineal respecto al hiperplano de
V03, y plausibilidad con N dev = 1483.

### F1. Random Fourier Features / Nystrom sobre la matriz exacta de V03 - **EJECUTAR, primera**

1. **Novedad.** 0 ocurrencias de `Random Fourier`, `random_fourier`, `rff`, `Nystrom`,
   `nystrom`, `skerf` y `nystroem` en los 156 scripts y 166 reports. Es genuinamente nueva.
2. **Complejidad.** D parametros por etiqueta sobre el mapa fijo, **independiente de N**.
   D = 2048 sobre 1731 filas son 28 MB en float64. Convexo, `liblinear`, segundos. Un
   barrido de 2 D x 3 gamma x 3 etiquetas son 18 ajustes y cabe en minutos. **Cabe en 32
   nucleos sin GPU** de sobra, y con el reparto exterior de V25 (8,77x) sobra tiempo para
   barridos grandes.
3. **Adecuacion a N = 1483.** Buena, y es su mejor argumento. El objeto es un problema
   convexo con penalizacion L2 y una unica palanca de capacidad, D. A 1187 filas por pliegue
   y D = 2048 el diseno esta levemente sobredeterminado, que es exactamente el regimen en el
   que L2 convexo esta bien condicionado. No hay superficie no convexa donde perderse.
4. **Mecanismo inductive bias (el punto que la hace no lineal de verdad).** RFF de kernel RBF
   sobre la matriz TF-IDF **sin reducir dimensionalidad**:
   `score = sum_d w_d cos(gamma * x . z_d + b_d) + b`. Ese `cos` expande en polinomios de
   `x . z`, o sea que el score incluye terminos de **interaccion de features ponderados por
   un kernel de similitud**, y ademas con la matriz completa de 27431 columnas el termino
   `(x . z_d)^2` es un **acuerdo de n-gramas casi coincidentes**, no un producto de dos
   columnas sueltas. El hiperplano de V03 es **puramente aditivo**: cuenta evidencia y no
   puede expresar "estos n-gramas aparecen juntos con estas frecuencias", que es lo unico que
   un `cos` con `gamma > 0` anade. En el limite `gamma -> 0` la funcion se aplana y el modelo
   degenera en V03, de modo que **la familia contiene a V03 como caso limite** y se puede
   verificar que la reproduce.
5. **Potencial de transferencia.** Es **la unica familia de la lista que ataca la causa
   diagnosticada de V23**. V23 perdio con 1321/1044/1317 vectores de soporte, o sea 60-76 %
   de las muestras: su capacidad no era declarada, era dependiente de la muestra, y por eso
   no se podia controlar ni medir. Un RFF **no tiene vectores de soporte por construccion**,
   o sea que el indicador que fundo a V23 es estructuralmente cero y D es una capacidad
   **declarada y monotona**. Es la unica forma de comprar la no linealidad del kernel sin
   pagar el precio que la pago. Y se aplica antes de cualquier reduccion, que es un punto
   distinto del espacio que V23 (SVD-300) y que kNN (2 espacios, ya cerrado con 93 % de
   redundancia).
6. **Coste de implementacion.** Bajo: una matriz de dibujo gaussiano y un `cos`. 40 lineas.
   El unico control que hay que escribir es que `gamma -> 0` reproduce V03.
7. **Riesgo especifico.** Que el `gamma` optimo **reproduzca la memorizacion de V23 con otro
   disfraz**: con `gamma` grande el kernel RBF se aproxima a la identidad y el clasificador se
   convierte en un vecino mas cercano, que es V23. Como la matriz es dispersa y desbalanceada
   eso es un riesgo real, no teorico. Mitigacion obligatoria: medir la **fraccion de
   memorizacion del vecino mas cercano** en dev y la **varianza entre semillas**, porque el
   RFF es una proyeccion aleatoria.
8. **Umbral de refutacion, en menos de una hora de CPU.** Barrido D en {1024, 4096} x gamma
   en {0,25, 1,0}, `C = 2.0` de V03, mismos pliegues, mismo recalculo canonico. Se cierra
   la familia si **ocurre cualquiera** de estas tres:
   - `max dOOF < +0.008` en las 8 celdas, o sea por debajo de V23 (+0.0132) y por encima de
     la banda que ya no distingue nada;
   - **fraccion de memorizacion del vecino mas cercano > 0,50** en la mejor celda, que
     seria la firma de V23;
   - **dispersion entre semillas > 0,003** de `dOOF`, que haria irreproducible el resultado.
9. **Recomendacion.** Ejecutar. Es la unica de las cuatro con un mecanismo que responde
   directamente al unico fallo medido del proyecto, y su costo es de minutos.

### F2. Tsetlin Machine (clausulas conjuntivas) implementado desde cero - **EJECUTAR, segunda, con tope**

1. **Novedad.** 0 ocurrencias de `Tsetlin`, `tsetlin`, `clause`, `Conjunctive`. Nueva.
2. **Complejidad.** El problema es la memoria, no el tiempo. A 27431 literales por 2000
   clausulas por etiqueta son 164M contadores, o sea 660 MB, **demasiado**. Hay que reducir
   el vocabulario a las 4000 columnas de mayor chi-cuadrado, lo que da 96 MB y cabe. El
   entrenamiento es de segundos sobre 1187 x 4000. **Cabe en 32 nucleos sin GPU.**
3. **Adecuacion a N = 1483.** **Debil, y hay que decirlo.** La literatura de Tsetlin machine
   rinde bien a partir de 10k-100k ejemplos cortos; con 1187 filas por pliegue el estimador de
   pesos de clausula esta hambriento de datos. Es la familia con la prior mas debil de las
   cuatro, y su valor no es competitivo sino **informativo**: es la unica que separa de forma
   limpia la pregunta "manda la clase de modelo" de la pregunta "manda la escasez de
   etiquetas". Si falla, y las otras tres son lineales o casi, la respuesta es que en este
   dataset manda la escasez.
4. **Mecanismo inductive bias.** Una clausula es una **conjuncion pura** de literales
   positivos y negativos, y dispara solo si se cumple la conjuncion. Eso es un sesgo de
   **emparejamiento de patrones / itemsets frecuentes** que un hiperplano **no puede
   expresses en absoluto**: V03 solo puede ordenar por suma de literales independientes, de
   modo que "el token A con el marcador de negacion B, y no el C" es representable como una
   clausula y **no** como una combinacion lineal de pesos, porque la decision depende de la
   presencia conjunta y no de la magnitud. Esa es la razon por la que no es "otro modelo
   lineal".
5. **Potencial de transferencia.** Correcto por construccion, y esto es una ventaja real: la
   complejidad crece con el numero de clausulas, que es un **parametro declarado y ajustable
   hacia abajo**, no con el numero de vectores de soporte. Si se reportan clausulas
   efectivas por etiqueta junto al resultado, la familia es auditable con el criterio de V24
   y V23, cosa que ninguna de las familias descartadas permite.
6. **Coste de implementacion.** El mas alto de las cuatro: el autmata de clausulas, la regla
   de feedback y el muestreo de literales, mas un control de correccion. Dos a tres horas
   para que sea confiable. Es el unico punto donde el dictamen puede fallar por coste.
7. **Riesgo especifico, y es grave: la binarizacion confunde clase de modelo con perdida de
   representacion.** Tsetlin machine necesita features binarias, o sea que **tira
   `sublinear_tf`, `min_df = 2` y el bloque `char_wb(3,5)`**, que es exactamente donde esta
   la ventaja de V03. Un resultado negativo no distinguiria "el modelo no sirve" de "le
   cortamos la representacion". Sin el control de la seccion 8 el experimento no es
   interpretable.
8. **Umbral de refutacion, con el control que rescata la interpretabilidad.** Se mide **el
   contraste dentro de la misma representacion**: LogisticRegression sobre las mismas 4000
   columnas binarizadas, y Tsetlin machine sobre ellas. Ademas el maximo absoluto. Se cierra
   si **ocurre cualquiera** de estas dos:
   - `dOOF(TM) - dOOF(LR sobre las mismas binarias) <= 0`, o sea la familia queda refutada
     **como clase de modelo** con independencia de la representacion;
   - `max dOOF < +0.005`, el nivel de la FM, que ya se sabe insuficiente.
   Y hay un tercer cierre, que **no es un resultado negativo**: si el autmata no converge en
   el presupuesto, se cierra por coste y se registra como **inconcluyente**, nunca como
   evidencia en contra.
9. **Recomendacion.** Ejecutar con tope de una tarde y el control dentro de la
   representacion. Sin ese control no se ejecuta.

### F3. Kernel de strings (subsequence / spectrum) como PRUEBA PREDICTIVA del mecanismo de V23 - **EJECUTAR, 20 minutos, no es candidata**

1. **Novedad.** 0 ocurrencias de `subsequence` y `spectrum`. Pero `kernel` aparece en 27
   archivos y SVC se uso en V23, asi que **la celda "kernel no lineal" ya esta ocupada y
   perdio**. Por eso esta familia se propone con un proposito distinto y mas valioso.
2. **Complejidad.** Matriz de Gram 1187 x 1187 por etiqueta, densa, 11 MB. Segundos.
   Trivial.
3. **Adecuacion a N = 1483.** La matriz de Gram de un kernel de substring sobre texto corto es
   **casi diagonal**: cada cadena solo se parece a si misma y a sus casi duplicados. Ese es el
   hecho que hace la prediccion del punto 4 testeable.
4. **Mecanismo inductive bias, y la prediccion falsable.** Sobre una Gram casi diagonal, el
   problema dual es tal que **casi todas las muestras se convierten en vectores de soporte**,
   o sea el modelo memoriza. Ese es el mecanismo de V23 (1321/1044/1317 de 1731) y aqui se
   puede **predecir a priori**, sin entrenar, a partir de la geometria de la matriz: si la
   similitud del vecino mas cercano por encima del diagonal es alta, el kernel memoriza.
   Esto **convierte la explicacion post-hoc de V23 en una prediccion puesta a prueba**, que
   es lo unico que puede subir de categoria, de "una explicacion de un dato" a "una teoria del
   proyecto". Y si la prediccion **falla**, la auditoria de V23 esta equivocada y hay que
   reabrir la direccion no lineal entera.
5. **Potencial de transferencia.** Nulo como candidata, y por eso no se propone enviar. Como
   **experimento de refutacion** es el mas barato y el mas informativo del lote: decide si las
   3 familias de kernel quedan cerradas por teoria y no por un solo caso.
6. **Coste de implementacion.** Nulo: un kernel de subsecuencia se escribe en 25 lineas.
7. **Riesgo especifico.** Que se lea como candidata y se acabe enviando. Prohibido por
   construccion: **esta familia no puede generar submission**, y el registro de gate lo dice.
8. **Umbral de refutacion.** Se construye la Gram de un solo pliegue, una etiqueta, tres
   nucleos de longitudes 2/3/4, y se mide la **mediana de la similitud del vecino mas
   cercano excluyendo el diagonal**. Si esa mediana es **> 0,60** para 2 de los 3 nucleos, la
   prediccion de memorizacion se confirma, la teoria de V23 queda **generalizada** y la
   familia entera de kernels no lineales sobre strings queda cerrada **por teoria**. Si la
   mediana es **< 0,30**, la teoria de V23 es incorrecta, la direccion no lineal se reabre y
   el dictamen de cierre queda **invalidado en su premisa**. La zona intermedia es
   inconclusa y no se puede usar para cerrar.
9. **Recomendacion.** Ejecutar. Es el experimento de mayor relacion valor/costo de toda la
   fase y **no cuesta ni una hora ni una submission**. Es tambien el unico que puede
   refutar este mismo dictamen, que es lo que se le pide a un dictamen serious.

### F4. Propagacion de etiquetas transductiva / regularizacion de manifold - **BLOQUEADA, ejecutar solo con autorizacion de reglas**

1. **Novedad.** 0 ocurrencias de `LabelPropagation`, `LabelSpreading`,
   `label_propagation`. Nueva. `propagacion` aparece en 18 archivos pero en otro sentido
   (propagacion de umbrales y de canales), no de etiquetas.
2. **Complejidad.** Sistema **lineal** `(I - alpha * S) f = y` con `f` multidimensional, una
   vez por etiqueta. `S` rala o densa de 1731 x 1731. Segundos. Sin GPU, sin ajuste de
   hiperparametros costoso. La unica palanca es `alpha` y el `k` del vecindario.
3. **Adecuacion a N = 1483.** La mas adecuada en principio, y por una razon concreta: **no
   aporta ningun parametro nuevo**. La funcion armonica hereda los scores de V03 como termino
   fuente y solo los suaviza sobre la geometria de la muestra. Con 1187 filas el
   smoothing es estable y el unico riesgo es el over-smoothing, que es visible como
   colapso de la tasa de positivos predicha.
4. **Mecanismo inductive bias.** Un **extremo armonico sobre un grafo de similitud**: la
   decision depende de la posicion en la nube de puntos, no de un hiperplano, y es por
   definicion no representable como `w . x + b`. Usa informacion que V03 **literalmente no
   puede acceder**: la geometria de las 1154 filas **no etiquetadas** del test, que es el
   unico activo del proyecto que nadie ha tocando.
5. **Potencial de transferencia.** Su unico efecto posible es **suavizar** un modelo de
   0.777, no anadir evidencia. Techo esperado bajo, y hay que decirlo: no es una candidata
   con probabilidad de ganar.
6. **Coste de implementacion.** Bajo, 30 lineas de forma cerrada.
7. **Riesgo especifico, y es de cumplimiento, no de modelado.** El encargo lista
   **pseudo-labeling como prohibicion dura**. El self-learning **esta prohibido de plano** y
   no se propone. La propagacion de etiquetas es tecnicamente distinta: no crea ninguna
   etiqueta dura, no hay argmax sobre etiquetas, y la solucion es un minimo convexo de
   forma cerrada cuyo termino fuente es V03. **Pero esa distincion es fragile frente a una
   lectura de las reglas**, y este dictamen no tiene autoridad para declararla. Poner features
   no etiquetadas del test en la construccion del grafo puede o no considerarse permitido.
7b. **Riesgo de segundo tipo: el sobre-smoothing** desplaza la tasa de positivos predicha, y la curva
   del LB contra la tasa esta medida con un maximo en V03 y con pendiente **-0,25 de LB por
   unidad de tasa** a la derecha, o sea que un desplazamiento pequeno cuesta LB rapido.
8. **Umbral de refutacion.** Barrido de `alpha` en {0,1, 0,3, 0,6, 0,9} con `k` en {5, 15},
   profile de celdas y variacion de la tasa predicha. Se descarta si la tasa predicha se mueve
   mas de **0,01** en cualquier celda (fuera de la banda medible del LB) o si
   `max dOOF < +0,005`.
9. **Recomendacion.** **No ejecutar hasta que la auditoria de reglas emita un veredicto
   escrito.** Condicion de apertura: un documento que diga explicitamente que la propagacion
   transductiva sobre features no etiquetadas, sin creacion de etiquetas, no es
   pseudo-labeling. Sin ese documento, la familia se descarta **sin gastar CPU**, y eso es
   una decision de cumplimiento, no de modelado.

## 5. Por que ningun gate autoriza una submission

El gate vigente de V24 (dOOF > +0.0376 y > 200 celdas) es incoherente, y la segunda parte ya
se sabe insuficiente: la FM cambio 278 celdas, paso el filtro, y tuvo **neto negativo (-40)**
con `dOOF` positivo. Un gate que acepta candidatos con neto negativo no es un gate.

**Gate corregido que se propone para cualquier fase futura**, con las cuatro condiciones
conjuntas:

1. `dOOF >= +0.024`, o sea 2 desviaciones tipicas locales, y no el +0.0376 puntual;
2. **neto de celdas > 0** y `|neto| >= 107` (la constante medida de 2,8e-05 de F1 de LB por
   acierto neto), porque sin las dos a la vez el signo de `dOOF` no significa nada;
3. indicador de complejidad **declarado y acotado** en el informe, con la direccion de V23
   (memorizacion dependiente de la muestra) medida y no supuesto;
4. **control de permutacion**: el mismo pipeline con etiquetas barajadas, para demostrar que
   el `dOOF` no es ruido de la superficie plana en la que vive V03.

**Y el resultado que cierra la question.** Aplicando el modelo de transferencia al mejor
caso que el gate permite, `dOOF = +0.024` da
`dLB = -0.00920 + 0.2448 x 0.024 = -0.00333`. O sea **el mejor candidato que el gate
autorizaria tiene un valor esperado negativo en el leaderboard**. Un gate cuyo mejor caso
tiene `dLB` esperado negativo no puede autorizar nunca una submission, por construccion, y
por eso la conclusion no es "no encontramos nada": es que **la herramienta de seleccion es
mas pequena que el efecto que se busca perseguir**.

Y hay una segunda razon, independiente y mas fuerte: la respuesta del LB a la tasa predicha,
que es la unica palanca que el proyecto ha [**causalmente**] manipulado (v02 -> v03, mismo
modelo, cambio de umbrales, +0,00313), tiene un recorrido total de **0,00811** entre sus
extremos, contra una desviacion tipica del LB de **0,01219**. **Todo el recorrido de la
curva cabe en menos de un error estandar.** El maximo aparente en la tasa de V03 no es un
pico, es ruido. Optimizar el punto de operacion no es una reserva, es una trampa, y es
precisamente la trampa en la que caiu V05, que tuvo el mejor local de las 22 y perdio 0,00704.

## 6. Orden de apertura y cierre

Regla de governance, para no abrir una cadena indefinida ni cerrar por un resultado unico.

| orden | familia | presupuesto | se cierra si | submit |
|---|---|---|---|---|
| 0 | F3 kernel de strings (prediccion) | 20 min | la mediana de similitud del vecino > 0,60 (confirma) o < 0,30 (reabre) | nunca |
| 1 | F1 RFF / Nystrom | 1 h | `dOOF < +0.008`, o memorizacion > 0,50, o dispersion entre semillas > 0,003 | solo si pasa el gate de la seccion 5, y el gate no lo pasa |
| 2 | F2 Tsetlin machine | 1 tarde | `TM - LR <= 0` dentro de la representacion, o `dOOF < +0.005`, o no convergencia (cierre por coste, **inconcluyente**) | nunca |
| 3 | F4 propagacion transductiva | bloqueada | sin veredicto escrito de reglas | nunca |

**Regla de cierre transversal, y es la que decide el proyecto.** Una familia se cierra con
**un** resultado negativo solo si ademas se cumple una de estas dos: (a) el resultado esta
**dentro del ruido del instrumento**, o sea `dOOF` en la banda +-0,003 alrededor de 0, o (b)
hay un **mecanismo medido** que lo explica, como el de F1 (memorizacion) o el de F2
(perdida de representacion). Un `dOOF` negativo sin las dos condiciones **no cierra nada** y
obliga a repetir con semilla distinta. Y al reves: **ningun resultado positivo abre una
submission** sin las cuatro condiciones de la seccion 5.

**Ultima observacion, y es la que mas pesa.** F3 se ejecuta **antes** que F1 y F2, y no para
ganar, sino porque puede **refutar este mismo dictamen**. Si la mediana de similitud del
vecino mas cercano en un kernel de substring cae por debajo de 0,30, entonces la explicacion
de V23 es incorrecta, la direccion no lineal no esta cerrada, y la conclusion de este
documento se cae. Un dictamen que no puede ser refutado por su propio primer experimento no
es un dictamen, es una opinion; este si lo es, y por eso F3 va primero.

## 7. Veredicto

El techo de `dOOF = +0.03756` no es alcanzable en este dataset. No porque la competencia sea
imposible, sino por tres medidas independientes y convergentes: su banda de confianza spans
un factor 6,8; el mejor `dOOF` del proyecto es el 9-61 % de lo necesario segun esa banda; y
la tasa de intercambio medida en la unica sonda con `dLB` emparejado (V23) exige un `dOOF` de
+0,0440. La hipotesis del lider de equipolvante es correcta y su consecuencia es contraintuitiva:
**V03 no tiene habilidad medible por encima del campo** (z = +1,557 contra un maximo esperado
de +1,494, `P = 0,575` bajo nueve sorteos puros), de modo que el liston para superarlo es el
de un outlier de 1,56 desviaciones, y la mejora que hace falta, **+0,00526**, es **3,3 veces**
la mejor mejora realizada de la historia del proyecto. Ninguna de las cuatro familias
seleccionadas tiene mecanismo para entregar un salto de ese tamano, y el gate mas honesto que
se puede escribir autoriza, en su mejor caso, un `dLB` esperado de **-0,0033**.

Las 3 submissions restantes **no se gastan** y quedan reservadas para una revision del
enunciado o de las reglas, que es la unica via por la que la conclusion de este dictamen
puede cambiar.

**NO HAY VIA COMPETITIVA**
