# P83: pre-registro del modelo bayesiano, con doble objetivo

Fecha de escritura: 2026-09-30
Proyecto: /home/carlos/Git/PLN-v67/kaggle_toxic_2026_puj
Rama: proyecto-kaggle-toxic-2026-p67
Script previsto: experiments/p83_01_bayesiano.py
Salidas previstas: reports/p83_bayesiano.json y reports/p83_bayesiano.md
Estado: PRE-REGISTRADO. Este texto se hashea y se commitea ANTES de escribir una linea de
codigo. Sin ese commit no hay experimento, igual que en P80, en P81 y en P82.

## 0. Decision de Carlos, y por que este documento se atreve a decir que las dos preguntas
   caben

El plan maestro, bitacora de las Fases 6 a 8:

> P83, modelo bayesiano. DECISION: doble objetivo, incertidumbre y calibracion, y canal de
> ensamblaje. Por eso hay que medir dos cosas distintas y no confundirlas: log-loss, Brier y
> curvas de calibracion por un lado, y perfil de error y complementariedad con V03 por otro.

La instruccion que recibio este documento pedia, textualmente: "Si el pre-registro no puede
sostener la doble afirmacion, dimelo y escribe solo la que se pueda sostener. Es preferible
un pre-registro con una pregunta que dos que no cierran."

Respuesta, mas exactamente: se sostiene la doble, y no porque sea mas generoso, sino porque
las dos preguntas se pueden aislar con reglas que tienen dientes, escritas en 3.

Respuesta, despues de revisar si las dos cierran: CIERRAN, y cierran porque son disjuntas por
construccion, no por cortesia. Se sostiene la doble afirmacion y por eso este documento
tiene las dos. Lo que hace falta para que cierren no es recortarlas, es separarlas con
reglas que tengan dientes, y esas reglas estan en la seccion 3. Si en la ejecucion apareciera
una contamination, la segunda se cae y se dice cual se cae, con las reglas de 3.4 ya escritas
por adelantado.

El argumento de que cierran, en cuatro puntos:

1. Las dos tienen metricas DISJUNTAS. La pregunta 1 se responde con log-loss, Brier, curvas
   de calibracion y cobertura de intervalos. La pregunta 2 se responde con un perfil de error
   y con la matriz de interseccion contra V03. Ninguna metrica de la pregunta 1 aparece como
   criterio de la pregunta 2, y ninguna metrica de la pregunta 2 aparece como criterio de la
   pregunta 1.
2. Los dos criterios tienen naturaleza DIFERENTE Y AMBOS ESTAN FIJADOS ANTES. La pregunta 1 no
   compite por F1 y su texto lo dice de entrada. La pregunta 2 no busca superar a V03 y su
   texto lo dice de entrada. No hay un unico numero que ambas cosas puedan mover, luego no
   pueden competir.
3. El unico objeto que comparten son las predicciones OUT-OF-FOLD, y ese objeto se genera
   UNA vez, con un unico protocolo, y no se puede reentrenar. La regla 3.5 prohibe
   explicitamente que el ensamblaje de la pregunta 2 alimente la calibracion de la pregunta 1
   y viceversa, porque las dos cosas que se contaminarian son el calibrador y el peso del
   ensamblaje, y ambas se fijan sobre el mismo OOF.
4. El riesgo real de contamination esta identificado, es concreto, y tiene regla: el
   calibrador. P76 YA calibro V03 y la TextCNN con un protocolo anidado, y por eso sabemos
   que el camino existe. Si P83 calibrara un miembro sobre el OOF completo y luego midiera su
   calibracion sobre ese mismo OOF, la pregunta 1 seria circular. La regla 3.5 lo prohibe y
   A14 lo verifica por instrumentacion.

## 1. Pregunta 1: incertidumbre y calibracion. NO compite por F1

Se dice de entrada, y se repite en el informe que P83 produce:

    LA_PREGUNTA_UNO_NO_COMPITE_POR_F1_Y_NO_ACE_POR_ELIMINADO_EL_ORDENAMIENTO_POR_F1

### 1.1 Enunciado

> Siendo V03 el modelo de referencia, un modelo bayesiano disperso sobre la misma
> representacion TF-IDF de V03 entrega probabilidades cuya calibracion es, como minimo, no
> peor que la de V03 ya calibrado por Platt, y cuyo intervalo predictivo cubre la etiqueta
> observada con la frecuencia declarada?

### 1.2 Por que esa pregunta tiene un liston real, y no uno inventado

La referencia de la pregunta 1 NO es V03 crudo. V03 crudo ya se sabe que esta
sobrecalibrado en la direccion de confianza, y P76 lo midio:

| modelo | Brier macro | log-loss macro | pendiente HS | pendiente TR | pendiente AG |
|---|---|---|---|---|---|
| v03_raw | 0.125771 | 0.403583 | 1.311048 | 1.360647 | 1.298334 |
| v03_cal | 0.122455 | 0.391009 | 0.996569 | 0.998942 | 0.974155 |

Una pendiente de 1.31 quiere decir que las probabilidades de V03 son demasiado extremas. Y
el paso de Platt las corrigio y mejoro el log-loss en 0.012574 y el Brier en 0.003316.

Consecuencia: el liston de la pregunta 1 es v03_cal, no v03_raw, y eso hace la pregunta
DIFICIL y por tanto valiosa. Un bayesiano que solo iguala a V03 crudo no ha hecho nada, y
eso hay que decirlo antes de correr, no despues.

### 1.3 Las metricas de la pregunta 1, y el margen de no inferioridad

Las cuatro, todas sobre el OOF pareado, sobre las 4449 unidades tweet-etiqueta, por etiqueta
y macro:

1. log-loss macro. Se declara el recorte, porque sin el el log-loss es inestable: recorte
   declarado en 1e-15, y se publica tambien el log-loss SIN recortar como cifra de contexto.
2. Brier macro, en su forma multiclase, la misma que la de P76 para que las cifras sean
   comparables.
3. Curva de calibracion, con 10 bins de igual anchura, por etiqueta y macro, con el numero
   de unidades por bin, con la frecuencia observada y la media predicha en cada bin, y con
   el area entre la curva y la diagonal, es decir el ECE ponderado. Se publica la curva
   entera, no solo el ECE, porque un ECE puede ser igual con dos curvas opuestas.
4. Cobertura de intervalos. Se define en 1.4 y lleva su control.

Margenes, fijados antes de ejecutar:

| Magnitud | Margen de no inferioridad | Justificacion del numero |
|---|---|---|
| log-loss macro | 0.005 absoluto | Es menos de la mitad de lo que la calibracion de Platt compro a V03 crudo, que fue 0.012574. Un bayesiano tiene que quedar a menos de 0.005 del Platt para poder declarar que no es peor. |
| Brier macro | 0.0015 absoluto | Es menos de la mitad de lo que Platt compro, que fue 0.003316. |
| ECE macro | 0.010 absoluto | Es aproximadamente el doble de la anchura de un bin de la curva, es decir, la resolucion misma del ECE. Pedir mas seria pedir mas que la resolucion de la medida. |

La lectura se hace con una regla de dos intervalos, escrita en 1.5, porque un solo intervalo
sobre cinco pliegues no alcanza.

### 1.4 Los intervalos y su control, que es la parte que hace que la pregunta valga

Objeto: para cada unidad tweet-etiqueta, el intervalo predictivo bayesiano al 90 por
ciento, es decir el intervalo de credibilidad al 90 por ciento de la probabilidad predictiva
posterior, con los cuantiles 0.05 y 0.95 de la predictiva posterior.

Medida de cobertura: la fraccion de unidades cuya etiqueta observada, 0 o 1, CAE DENTRO del
intervalo. Se declara una salvedad honesta ANTES de ejecutar: para una variable binaria, la
probabilidad de que un intervalo de credibilidad al 90 por ciento contenga la etiqueta es
alta por construccion, y la metrica esta sesgada al alza. Por eso el numero de 0.90 NO es un
liston, y por eso la metrica de verdad no es la cobertura sola sino la ANCHURA, comparada
contra un control degenerado.

Control degenerado, obligatorio, y es la parte que hace que la afirmacion tenga poder
discriminante:

- Se construye el intervalo de anchura CERO centrado en la probabilidad de V03, es decir, el
  intervalo igual a [p_V03, p_V03]. Su cobertura es exactamente la exactitud de V03.
- Un bayesiano con un intervalo estrecho que no cubre va a perder contra ese control. Un
  bayesiano con un intervalo absurdamente ancho, el intervalo completo [0, 1], tiene
  cobertura 1.0 y no dice nada. Los dos extremos estan acorralados, y por eso la metrica
  tiene que ser la cobertura CON penalizacion por anchura, que se declara:
  cobertura_menos_pesada_por_anchura, igual a cobertura del bayesiano menos 0.5 por la
  anchura media normalizada de su intervalo.
- Se publica ademas, y sin ocultarlo, el numero de veces que la etiqueta observada cae
  en el intervalo de V03 de anchura cero, que es la exactitud de V03, para que se vea
  cuanto vale la cobertura de referencia.

Lo que P83 autoriza sobre incertidumbre: o el intervalo bayesiano cubre al menos tanto como
el intervalo de V03 Y es mas estrecho, o no cubre, o es mas ancho. Lo que no autoriza: decir
que el modelo "sabe mas", porque la cobertura no mide saber.

### 1.5 La regla de lectura de la pregunta 1, fijada antes de ejecutar

Sean I_f el intervalo t pareado de los cinco deltas por PLIEGUE, y I_s el intervalo t
pareado de los deltas por SEMILLA, con el pooled over 25 unidades correlacionadas solo si se
declara que esas unidades no son independientes, y se declara: NO SON INDEPENDIENTES. Los
cinco pliegues particionan las mismas 1483 filas y cada una se repite en cinco semillas, luego
son veinte y cinco unidades altamente correlacionadas sobre un unico conjunto. Es exactamente
el punto que P78 escribio sobre sus "5 de 5 pliegues" y aqui se aplica al bayesiano.

Consecuencia, escrita: los intervalos por pliegue y por semilla NO son intervalos
independientes y no se puede declarar un nivel de confianza del 95 por ciento sobre ninguno de los dos. Se publican los dos, con su etiqueta, y la regla de lectura es:

- AFFIRMATIVO: los DOS intervalos, el de pliegues y el de semillas, tienen su extremo en la
  direccion favorable mas alla del margen de 1.3. O sea, para log-loss, el extremo superior
  del intervalo del delta queda por debajo de menos 0.005, es decir, por debajo de menos el
  margen.
- NO INFERIOR, SIN AFIRMAR SUPERIORIDAD: los dos intervalos contienen el margen, o uno lo
  contiene y el otro no, pero ninguno lo cruza hacia el lado desfavorable.
- NEGATIVO: los dos intervalos tienen su extremo en la direccion desfavorable mas alla del
  margen.
- AMBIGUO Y POR LO TANTO NADA: los dos intervalos se contradicen en el signo, o uno excluye
  el margen por el lado favorable y el otro lo cruza por el desfavorable. En ese caso no se
  afirma nada y se publica el caso con sus numeros. Se declara en el informe que el numero de
  pliegues es cinco y que un intervalo de cinco observaciones es fragil por construccion.

## 2. Pregunta 2: complementariedad de errores con V03. NO busca superar a V03

Se dice de entrada:

    LA_PREGUNTA_DOS_NO_BUSCA_SUPERAR_A_V03_SU_CRITERIO_ES_TENER_UN_PERFIL_DE_ERROR
    DISTINTO_Y_MEDIBLE

### 2.1 Enunciado

> Tomando las predicciones OUT-OF-FOLD de V03 como referencia, el miembro bayesiano falla
> en un subconjunto de filas distinto, de forma medible y por encima del azar, tal que su
> incorporacion al ensamblaje cambia el numero de decisiones acertadas en celdas donde V03
> falla?

### 2.2 El criterio, y por que no es "ganar a V03"

El criterio de admision como miembro de ensamblaje tiene TRES condiciones, todas medidas
sobre el OOF y NUNCA sobre el holdout:

- C1, DIVERSIDAD MEDIBLE: el AUC de la probabilidad del propio miembro como predictor del
  error de V03, calculado sobre el OOF, es mayor o igual que 0.55 en al menos DOS de las
  tres etiquetas, o su acuerdo con V03 es menor o igual que 0.80.
- C2, RESCATE REAL: el rescate del ensamblaje, es decir las unidades en las que el ensemble
  acierta y V03 falla, es mayor que cero, y mayor que el dano, sobre el OOF, en al menos uno
  de los tres alphas declarados.
- C3, NO PERJUDICA A V03: el corrected_cv del ensamblaje con el ALPHA PRIMARIO no es inferior
  al de V03 en mas de 0.003, que es el EPS_GATE_DELTA que uso P77 y P78, con los tres
  umbrales de V03 congelados.

El criterio NO es que el miembro supere a V03. La razon se escribe porque el historial la
impone:

- El NB-SVM canonico ya dio 0.763839, con 0 de 5 pliegues a favor, es decir -0.013372
  respecto a V03. Y no es un caso raro: "ni siquiera con umbral optimo el canonico alcanza al
  campeon", con 0.767669 en la muestra, y eso es un numero optimista que no es publicable.
- La regresion logistica L1 con C igual a 4.0 dio 0.783423, es decir +0.006211 y 5 de 5
  pliegues, PERO con el intervalo pareado IC95 en (-0.0081, +0.0202), que CONTIENE EL CERO.
  Es decir, la candidata numero uno de la priorizacion de este proyecto, la que mas senal
  consistente habia mostrado, no se distingue de la razon por la que se eligio.
- El precedente de ensamblaje tambien es negativo: P75 probo TextCNN mas V03 y el veredicto
  fue CNN_PERJUDICA, con rescate 17 y dano 21 en la mejor configuracion, y el corrected_cv
  cayo en todos los ocho casos de la matriz de P76.

Por eso la expectativa declarada del P83, escrita ANTES de correr, es esta:

    CON_1483_FILAS_Y_CELDAS_RARISIMAS_EL_PAPEL_ESPERADO_DEL_BAYESIANO_ES_MIEMBRO_DIVERSO
    NO_CAMPEON

Y por eso la pregunta 2 se declara EXPLORATORIA en su naturaleza, igual que la 1, y su
resultado no puede producir ninguna afirmacion sobre el leaderboard, ni aunque el
corrected_cv del ensamblaje suba.

### 2.3 Por que se toca el holdout jamas, ni siquiera para esto

El rescate y el dano se calculan sobre el OOF pareado, que son las predicciones de filas que
el modelo NO vio al entrenar. El holdout son 248 filas congeladas, y dl-specialist ya senalo
que DL-03 compromised el presupuesto del holdout, que era de cinco evaluaciones, al entrenar
sobre 1731 filas. P83 no gasta ninguna de esas cinco, y A1 lo verifica.

## 3. Las reglas de separacion, que son las que sostienen el "CIERRAN" de 0

### 3.1 Que metricas van a cual bloque, y que no se mezclan

| Bloque | Metricas | Prohibido usar |
|---|---|---|
| PREGUNTA 1 | log-loss, Brier, curva de calibracion, cobertura, ECE, anchura de intervalo | f1_media_columnas, corrected_cv, rescate, dano, AUC de error |
| PREGUNTA 2 | AUC de error de V03, acuerdo, matriz de interseccion, rescate, dano, corrected_cv del ensamblaje con umbrales congelados | log-loss, Brier, ECE, cobertura |

El f1_media_columnas del miembro bayesiano se PUBLICA, porque no publicar un numero que se
va a calcular es peor que publicarlo mal. Pero se publica como CIFRA DESCRIPTIVA, con la
etiqueta DESCRIPTIVA_Y_SIN_CUALQUIER_LECTURA_INFERENCIAL, y jamas aparece como criterio de
ninguna de las dos preguntas. Ni como criterio de la 1, que no compite por F1, ni como
criterio de la 2, que no busca ganar.

### 3.2 El V03 de referencia es UNO SOLO y es el OOF canonico

- V03 es FROZEN. No se reentrena, no se toca, no se consulta el leaderboard.
- Sus umbrales 0.44, 0.51 y 0.46 se leen del artefacto congelado y se usan sin reoptimizar.
  El numero de reoptimizaciones sobre cualquier conjunto tiene que ser 0, y A2 lo verifica.
- Las predicciones de referencia son el OOF canonico de V03, el mismo que usaron P75 y P76,
  y su corrected_cv es 0.7772114525, que se recalcula y se compara dentro de 1e-9. Si no
  coincide, STOP, porque significaria que el OOF de referencia no es el de siempre y todas
  las cifras de complementariedad serian in comparables con P75.

### 3.3 El calibrador se ajusta en datos internos, nunca en el OOF

Es la regla que sostiene la honestidad de la pregunta 1, y es la que P76 ya demostro que se
puede hacer:

    outer_train -> inner_train + inner_calibration -> calibrador -> outer_validation
    la outer_validation permanece COMPLETAMENTE aislada

Consecuencia declarada: las cifras de la pregunta 1 son CONDICIONALES a que el modelo se
ajuste en el inner_train y el calibrador en el inner_calibration, y son comparables con las
de P76 con la asimetria que P76 declaro, a saber: el calibrador se ajusta sobre unas 830
filas con modelos entrenados en el inner_train, y se aplica a las predicciones que vienen de
modelos entrenados sobre el outer_train completo de 1186. Esa asimetria se declara en el
informe de P83, con la misma frase con la que la declaro P76, porque es la misma
asimetria.

Lo que P83 NO hace: ajustar Platt ni isotonic ni temperatura sobre el OOF completo y luego
medir la calibracion sobre ese mismo OOF. Eso daria una curva de calibracion inflada y todo
el bloque 1 seria circular. A14 lo verifica por instrumentacion y por mutacion.

### 3.4 Que se cae si hay contamination, decidido por adelantado

Si al ejecutar P83 resultara que las dos preguntas no pueden sostenerse separadas, las
reglas son estas y estan fijadas ANTES:

- Si el OOF del miembro bayesiano no se puede generar sin tocar el holdout, se cae la
  PREGUNTA 2 entera, porque la 2 es la que necesita el pool grande. La 1 se sostiene con
  pliegues sobre desarrollo.
- Si el calibrador solo se puede ajustar sobre el OOF, se cae la PREGUNTA 1, porque medir
  calibracion sobre los datos con los que se calibro no es medir. La 2 no depende de la
  calibracion y se sostiene sola.
- Si las dos se caen, no se escribe un informe de P83. Se escribe un documento corto que diga
  que la doble pregunta no se podia sostener con estos datos, y se publica como tal.
- Ninguna de las tres salidaes permite publicar un numero de F1 como si fuera un resultado.

### 3.5 Las prohibiciones de contaminacion, escritas para que se puedan auditar

- El peso del ensamblaje, el alpha y cualquier parametro de la calibracion se fijan sobre
  datos INTERNOS de cada pliegue. Nunca sobre el OOF completo.
- El alpha del ensamblaje se declara aqui, ANTES de ejecutar, y es una rejilla con un
  primario, igual que en P75. Rejilla: 0.05, 0.10 y 0.15. ALPHA PRIMARIO: 0.10, que es el
  que cargo la puerta en P75. Los otros dos se publican completos, con sus numeros, sin
  promoverlos y sin rescatar un fallo del primario. Se publica tambien la lectura con
  reoptimizacion del alpha sobre el OOF, marcada como DIAGNOSTICA_Y_SIN_LECTURA_INFERENCIAL,
  porque omitirla seria ocultar la lectura que mas favorece a la conclusion.
- No se usa un mismo OOF para elegir el modelo, ajustar el calibrador y medir la
  calibracion. Y esto se verifica con contadores, no con confianza.
- Lo que se publica del bloque 2 se etiqueta como DESCRIPTIVO. Si el ensamblaje sube el
  corrected_cv sobre el OOF, eso NO es una mejora de V03, es una lectura de un ensayuelo
  sobre datos que ya se usaron para elegirlo, y el informe tiene que decirlo con esas
  palabras.

## 4. Que se mide ya, sin modelo, y con que numeros

Contexto ya publicado por el proyecto, que se cita y no se recalcula:

- V03 corrected_cv: 0.7772114525
- NB-SVM canonico: 0.763839, con 0 de 5 pliegues a favor, -0.013372
- Logistica L1 con C igual a 4.0: 0.783423, +0.006211, 5 de 5 pliegues, IC95 pareado
  (-0.0081, +0.0202) que CONTIENE EL CERO
- v03_raw y v03_cal: la tabla de 1.2

Y la caracterizacion del problema, que es la que justifica el metodo bayesiano y su limite:

- 1483 filas de desarrollo, 1645 grupos, 5 pliegues agrupados de 347, 346, 346, 346 y 346.
- TF-IDF de V03 con sublinear_tf igual a True, luego la matriz es rala y con MUCHAS celdas
  que solo aparecen una vez. Con 1483 filas, "muchas celdas rarisimas" es la descripcion
  correcta, y es la condicion en la que un metodo disperso con seleccion automatica de
  variables tiene sentido y uno ridge con todos los coeficientes no.
- El pre-registro de P78 y el de P82 cuentan las historias de contaminacion por checkpoint.
  Aca el riesgo equivalente es la seleccion de variables, y por eso la
  seleccion de variables se hace SOLO sobre el inner_train, nunca sobre el OOF.

## 5. El metodo bayesiano, declarado con su aproximacion

### 5.1 Brazo M1, la logistica bayesiana DISPERSA. Aproximada, y se dice

- Modelo: regresion logistica por etiqueta, una etiqueta por vez, sobre la TF-IDF de V03.
  Es un producto de tres binarias independientes a nivel de parametro y no de etiqueta, y
  esa independencia es una APROXIMACION que se declara. HatEval Task B oficial reporta ademas
  Exact Match Ratio, luego la equivalencia exacta de la metrica local con la del host no
  esta confirmada, y eso ya lo declaro P78.
- Prior: NORMAL con media cero y varianza(alpha_k) inversa, con un unico alpha_0 declarado
  igual a 1.0. La dispersion por coeficiente es lo que hace el modelo disperso: un
  coeficiente con alpha grande, es decir varianza pequena, se ancla a cero, y un coeficiente
  con alpha pequeno queda libre.
- Aproximacion de la posterior: INFERENCE VARIACIONAL DE CAMPO MEDIO, con la posterior
  aproximada como un producto de gaussianas independientes por coeficiente, con media m_k y
  varianza s_k. La moda de la posterior se calcula con IRLS o con Newton, y el update de la
  varianza es el de MacKay, es decir gamma_k igual a 1 menos m_k al cuadrado por 1 menos
  gamma_k. El numero de iteraciones del punto de variacion se declara en 30, con tolerancia
  1e-6 sobre la varianza de la posterior.
- Referencias, verificables, que fijan la convencion y no el resultado:
  - Michael E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine, Journal of
    Machine Learning Research 2, 2001, paginas 226 a 253. Es la fuente de la actualizacion de
    la varianza de campo medio y de la nocion de numero efectivo de parametros.
  - Yuanqing Ng y Andrew Y. Ng, Adaptive Sparsity for Nonparametric Variable Selection,
    Journal of Machine Learning Research 13, 2012, paginas 1833 a 1861. Es la fuente de la
    determinacion adaptativa de la dispersion con priors heavy-tailed.
  - Christopher M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006, capitulo
    11, Variational Bayes. Es la fuente del update de MacKay y de la forma de la posterior
    aproximada.
- LO MAS IMPORTANTE Y LO QUE SE DECLARA DE ENTRADA: la prior y la posterior de este brazo
  son APROXIMADAS. No son la posterior exacta, son una aproximacion de campo medio. Por lo
  tanto los intervalos de credibilidad que salen de este brazo heredan el error de la
  aproximacion, y ese error NO SE CUANTIFICA. A15 es la afirmacion que intenta acotarlo, y su
  limite se escribe en 5.5.

### 5.2 El numero de muestras efectivo es un parametro del METODO, y no del DATO

El punto que el pre-registro tiene que hacer explicito, porque es la confusion mas facil que
hay con este metodo y porque se escribe en el informe con estas palabras:

    EL_NUMERO_DE_PARAMETROS_EFECTIVOS_QUE_PRODUCE_ESTE_METODO_ES_UN_PARAMETRO_DEL_METODO
    Y_ESTA_DETERMINADO_POR_EL_ESCALA_DE_LA_PRIOR_ALPHA_0_Y_POR_LA_VARIANZA_POSTERIOR
    NO_ES_UN_NUMERO_DE_MUESTRAS_EFECTIVAS_DEL_CONJUNTO_DE_DATOS

El numero de grados de libertad efectivos es la suma de uno menos gamma_k sobre los
coeficientes. Es una consecuencia de la aproximacion y del prior, y varia si se cambia
alpha_0. NO es una propiedad del conjunto de 1483 filas. Publicarlo como "el conjunto de
datos tiene N grados de libertad efectivos" es un error de categoria, y queda prohibido.

Lo que se publica, y con que nombre, para que no haya ambiguedad:

- numero_de_parametros_efectivos, que es la suma de uno menos gamma_k, por etiqueta, por
  pliegue y por semilla.
- El contraste con un punto de referencia PUBLICO y NO AJENO: el numero de coeficientes con
  gamma_k mayor que 0.5, y el numero de variables que la regresion L1 con C igual a 4.0 dejo
  fuera, que es un dato ya medido en el proyecto. Si el bayesiano declara mas variables
  activas que la L1 y peor en F1, ese es un hallazgo sobre la diferencia de los dos criterios
  de seleccion, y no sobre los datos.
- Ademas, una rejeilla declarada de alpha_0 con 0.5, 1.0 y 2.0, y se publica el numero de
  parametros efectivos en CADA valor de la rejeilla, precisamente para DEMOSTRAR que ese
  numero se mueve cuando se cambia un parametro del metodo y no un parametro del dato. Si el
  numero de parametros efectivos no se mueve al cambiar alpha_0, entonces no se esta
  midiendo lo que se dice medir, y A16 falla.

### 5.3 Seleccion previa de variables, y su riesgo de leakage

La TF-IDF de V03 tiene muchisimas columnas y solo 1483 filas. Una regresion dispersa con ARD
sobre esa matriz es cara y numerically fragile. La solucion del proyecto es la seleccion
previa, y es la parte del diseno con mas riesgo de contaminacion:

- La seleccion se hace SOBRE EL INNER_TRAIN DE CADA PLIEGUE, y NUNCA sobre el OOF, nunca
  sobre el outer_train completo y nunca sobre el conjunto entero.
- La estadistica de seleccion es el producto punto del documento, es decir el valor del
  TF-IDF, y se exige que el valor sea mayor o igual que un UMBRAL FIJO declarado, igual a 3.
  Se declara que es una eleccion, no una optimalidad, y que un umbral mas alto reduce la
  dimension y puede perder senal.
- Se exigiran DOS sensibilidades del umbral, 2.0 y 5.0, con el 3.0 como primario, y las tres
  se publican. La dimension resultante de las tres se publica con cada pliegue, porque la
  dimension seleccionada es en si misma un resultado.
- Se publica el numero de columnas que sobreviven en cada pliegue, y se comprueba que es
  MENOR que el numero de filas del inner_train. Si en algun pliegue sobreviven mas columnas
  que filas, la regresion no va a estar determinada y hay que decirlo, y A17 exige que se
  diga.
- Riesgo que se declara: la seleccion por valor absoluto puede favorece a las columnas
  largas, luego la comparacion de dimensiones entre brazos hay que hacerse con
  esa salvedad. El metodo que sesga a favor del bayesiano es el que deja MAS columnas, luego
  un bayesiano mas disperso no gana por ganar: gana por el criterio de seleccion. A17 existe
  justo para que esa lectura este disponible.

### 5.4 El segundo brazo, y por que existe: un bayesiano EXACTO como referencia

Brazo M2: clasificador de Bayes de Bernoulli, o multinomial, con prior de Dirichlet sobre
los terminos, y verosimilitud conjuga. Es un bayesiano EXACTO, con forma cerrada y sin
ninguna aproximacion.

Por que existe, y no es por decoracion:

- Da un punto de bayesianidad EXACTA contra el cual comparar la aproximacion de M1. Si M1
  sale peor que M2 en calibracion, y no se puede saber si es porque la aproximacion es mala o
  porque el modelo es peor, porque no hay ningun bayesiano exacto de referencia.
- Es barato, y su forma cerrada se puede verificar con un segundo camino de calculo
  independiente. Es la unica afirmacion de este pre-registro que tiene un control aritmetico
  EXACTO, y eso tiene valor aparte.
- Su fallo esperado se declara por adelantado: MNB es probabilidades sobreconfidentes, luego
  su curva de calibracion va a estar peor que la de M1 y su intervalo va a ser mas estrecho
  de lo que la evidencia permite. Si M2 gana en F1 y pierde en calibracion, eso es un
  hallazgo que hay que publicar, y es exactamente la separacion de las dos preguntas.

### 5.5 El limite de 5.1, y la afirmacion que lo acota

El limite, escrito antes de ejecutar:

> El anclaje a una libreria de inferencia bayesiana queda SIN HACER. M1 se implementa desde
> cero en este proyecto.

Consecuencias:

- Lo que P83 autoriza es afirmar sobre el COMPORTAMIENTO del modelo que este proyecto
  implemento, y sobre la COMPARACION entre M1 y M2, que estan ambos implementados aqui.
- Lo que P83 NO autoriza es decir que la inferencia variacional de M1 es correcta, ni que
  su posterior aproximada se acerca a la exacta, ni comparar con pymc, con pyro, con
  sklearn, con edward o con cualquier otra libreria.
- Se comprueba primero si alguna de esas librerias esta en el venv. Si alguna esta, la
  afirmacion A15 la usa como ANCLA CONTRASTE y el limite se reduce. Si no esta, el limite se
  mantiene y se publica que no esta. La comprobacion se hace al escribir el informe, no se
  asume.
- Y se declara que la serie P80 y P81 tiene un precedente de ancla propia: P80 concluyo H1
  contra una reimplementacion propia, y ml-reviewer senalo que eso no demuestra lo que el
  titular afirmaba. P83 no cae en eso: su titular es sobre SU propio modelo y SU propia
  comparacion, y el limite de 5.5 es lo que impide que se lea como una afirmacion sobre
  bayesianidad en general.

## 6. Las diecinueve afirmaciones, indice

| Id | Bloque | Que hace | Naturaleza | STOP |
|---|---|---|---|---|
| A0 | 0 | sha256 del congelado, tamano y mtime antes y despues | CONFIRMAR | si |
| A1 | 0 | solo se leen las fuentes declaradas, 0 de test y 0 de holdout | CONFIRMAR | si |
| A2 | 0 | umbrales de V03 intactos, V03 no reentrenado, 0 reoptimizaciones | CONFIRMAR | si |
| A3 | 0 | el OOF de referencia es el canonico, coincide con 0.7772114525 | CONFIRMAR | si |
| A4 | 0 | la seleccion de variables se hace solo sobre el inner_train | CONFIRMAR | si |
| A5 | 1 | la posterior de M1 esta aproximada, y el metodo se declara | CONFIRMAR | si |
| A6 | 1 | el numero efectivo es un parametro del metodo, y se mueve con alpha_0 | CONFIRMAR | si |
| A7 | 1 | M2 es exacto y coincide con un segundo camino de calculo | CONFIRMAR | si |
| A8 | 1 | las seis magnitudes de la pregunta 1 medidas y publicadas | CONFIRMAR | no |
| A9 | 1 | la curva de calibracion completa, con bins poblados | CONFIRMAR | no |
| A10 | 1 | cobertura y anchura, y el control degenerado de V03 | CONFIRMAR | si |
| A11 | 1 | la regla de lectura de 1.5 aplicada, sin reinterpretacion | CONFIRMAR | si |
| A12 | 2 | las cuatro medidas de complementariedad contra V03 | CONFIRMAR | no |
| A13 | 2 | el rescate y el dano, sobre OOF, con los tres alphas | CONFIRMAR | no |
| A14 | 3 | el calibrador se ajusta en internal, verificado por contadores | CONFIRMAR | si |
| A15 | 3 | el limite de 5.5 verificado: anclaje externo presente o ausente | CONFIRMAR | no |
| A16 | 3 | la rejeilla de alpha_0 y la sensibilidad de umbral, publicadas | CONFIRMAR | no |
| A17 | 3 | dimension seleccionada por pliegue, y la salvedad de sesgo | CONFIRMAR | no |
| A18 | 4 | las cadenas de formulacion, de las dos preguntas y del proyecto | CONFIRMAR | si |
| A19 | 4 | el F1 del miembro se publica como descriptiva, sin lectura inferencial | CONFIRMAR | si |

Veinte filas contando A0: diecinueve afirmaciones de contenido mas la de integridad.
Dieciseis de CONFIRMAR con STOP, cuatro de CONFIRMAR sin STOP, cero REFUTAR.

## 7. Bloque 0: integridad, referencias y fuentes de contamination

**A0. Integridad de los artefactos congelados.**

- Se comprueba el sha256 de modelos/twitter_domain/w2v_twitter_100d.npz, que vale
  302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a, antes de la primera
  lectura y despues de la ultima, con size y mtime_ns. Se comprueba tambien el artefacto de
  V03, del que salen los umbrales congelados.
- Tolerancia: coincidencia exacta de los tres, antes y despues, y entre si.
- Exito: todos coinciden.
- Fallo: cualquiera difiere. STOP INMEDIATO con SystemExit y aviso a Carlos. No se registra
  ningun otro numero.
- STOP: si.

**A1. Solo las fuentes declaradas, y el holdout intacto.**

- Que se verifica: un contador de filas leidas por fuente, con el pool esperado de 1483
  filas de desarrollo, CERO filas de test, CERO filas de holdout, y que el total consumido de
  las cinco evaluaciones del holdout del proyecto sigue siendo cero, es decir, que P83 no
  anade ninguna.
- Tolerancia: enteros exactos. 0, 0 y 0.
- Exito: 1483, 0, 0.
- Fallo: cualquier lectura de test o del holdout, o cualquier evaluacion del holdout.
- STOP: si.

**A2. V03 intacto, umbrales intactos, cero reoptimizaciones.**

- Que se verifica: que los tres umbrales leidos del artefacto de V03 son 0.44, 0.51 y 0.46;
  que el numero de veces que P83 ajusta un umbral sobre cualquier conjunto es 0, con
  contador explicito; y que V03 no se reentrena ni se copia con parametros distintos.
- Tolerancia: igualdad exacta de los tres, contador de reoptimizacion igual a 0.
- Exito: los tres.
- Fallo: cualquiera.
- STOP: si.

**A3. El OOF de referencia es el canonico.**

- Que se verifica: que el OOF de V03 que usa P83 tiene el corrected_cv igual a
  0.7772114525, recalculado desde la tabla de predicciones y comparado con el publicado, con
  tolerancia 1e-9; y que el numero de unidades tweet-etiqueta es 4449.
- Por que es STOP: si el OOF de referencia no es el de siempre, las cifras de
  complementariedad de P83 no son comparables con las de P75, y toda la pregunta 2 se queda
  sin contra. A3 convierte esa comparabilidad en un hecho y no en una suposicion.
- Tolerancia: 1e-9 en el corrected_cv, entero exacto en las unidades.
- Exito: las dos.
- Fallo: cualquiera.
- STOP: si.

**A4. La seleccion de variables se hace solo sobre el inner_train.**

- Que se verifica, por instrumentacion: se parchea la funcion de seleccion y se registra de
  que indices de columna se la llama. Ninguna llamada puede recibir una mascara de columnas
  que incluya filas del outer_train completo, del outer_validation o de development entero.
  Se exige que TODAS las llamadas al selector usen exclusivamente filas del inner_train del
  pliegue y de la semilla que se esta procesando. Ademas, por mutacion, se comprueba que la
  comprobacion SABE FALLAR, ejecutandola una vez con una seleccion hecha sobre el
  outer_train completo: tiene que detectar la fuga.
- Tolerancia: conjuntos de columnas y de filas, identidad exacta. Cero filas fuera del
  inner_train.
- Exito: todas las llamadas dentro del inner_train, y la mutacion detectada.
- Fallo: cualquiera. La seleccion de variables sobre datos que despues se evaluan es la
  fuga equivalente a la del checkpoint, y es la que condemned P73 y P74 en otra forma.
- STOP: si.

## 8. Bloque 1: la pregunta de incertidumbre y calibracion

**A5. La posterior de M1 esta aproximada y el metodo se declara.**

- Que se verifica: que el informe declara el nombre del metodo, el tipo de prior, el tipo de
  aproximacion de la posterior, el numero de iteraciones del punto de variacion, la
  tolerancia, y que escribe la cadena literal
  LA_POSTERIOR_DE_M1_ES_APROXIMADA_Y_NO_ES_LA_POSTERIOR_EXACTA
- Tolerancia: coincidencia exacta de cadenas, y los seis campos presentes.
- Exito: los siete.
- Fallo: cualquiera. Si el informe describe la posterior como exacta, todo el bloque 1 es
  una lectura de un artefacto mal identificado, que es el fallo de P79 sobre el scheduler y
  de P80 sobre la matriz de salida.
- STOP: si.

**A6. El numero efectivo es un parametro del metodo, y se mueve cuando se cambia alpha_0.**

- Que se verifica: que el numero de parametros efectivos se publica para los TRES valores de
  alpha_0 de la rejeilla, 0.5, 1.0 y 2.0, por etiqueta y por pliegue; y que el numero en
  alpha_0 igual a 0.5 es DISTINTO del numero en alpha_0 igual a 2.0, con una diferencia
  relativa mayor que 0.05 en al menos una etiqueta y un pliegue; y que la cadena literal
  EL_NUMERO_DE_PARAMETROS_EFECTIVOS_QUE_PRODUCE_ESTE_METODO_ES_UN_PARAMETRO_DEL_METODO esta
  escrita en el informe.
- Por que se exige que SE MUEVA, y no que se publique: si el numero de parametros efectivos
  no responde a alpha_0, entonces no es el numero de parametros efectivos de la
  inferencia variacional, y el bloque 1 de la metrica de incertidumbre no tiene objeto. A6 es
  la afirmacion que impide que se publique un numero que no significa nada. Es la misma
  logica que la que exige que cada comprobacion sepa fallar.
- Tolerancia: diferencia relativa mayor que 0.05 entre los dos extremos de la rejeilla.
- Exito: los tres.
- Fallo: cualquiera.
- STOP: si.

**A7. M2 es exacto y coincide con un segundo camino de calculo.**

- Que se verifica: que la forma cerrada de M2 da la misma matriz de probabilidades que un
  segundo camino de calculo independiente, con la verosimilitud multinomial y el prior de
  Dirichlet calculados de otra forma, con tolerancia 1e-12 en la mayor diferencia absoluta
  entre dos probabilidades; y que el segundo camino se ejecuta de verdad, con su codigo, y no
  se afirma por construccion.
- Tolerancia: 1e-12 en diferencia absoluta de probabilidad. Es la unica afirmacion del
  proyecto con una tolerancia de ese orden, y es razonable porque las dos cuentas son
  exactas en aritmetica de punto flotante.
- Exito: la diferencia maxima menor o igual que 1e-12.
- Fallo: mayor, o el segundo camino no se ejecuto. Si no se puede reproducir una cuenta que es
  cerrada, algo se ha escrito mal y no se va a saber que.
- STOP: si.

**A8. Las seis magnitudes de la pregunta 1, medidas y publicadas.**

- Las seis: log-loss macro con recorte en 1e-15; log-loss macro sin recortar, como cifra de
  contexto; Brier macro en la forma de P76; Brier por etiqueta; ECE macro ponderado; y
  pendientes de la regresion probabilidad sobre etiqueta, por etiqueta, que es lo que P76
  publico y lo que hace comparables las cifras.
- Tolerancia: seis decimales. No hay cota de exito numerico, lo que falla es la medicion.
- Exito: las seis, por cada uno de los dos brazos, por cada pliegue y por el OOF completo.
- Fallo: falta alguna, o sale un NaN.
- STOP: no.

**A9. La curva de calibracion completa, con los bins poblados.**

- Que se publica: 10 bins de igual anchura, por etiqueta y macro, y por cada bin el numero de
  unidades, la frecuencia observada y la media predicha. Se exige que se publique la curva
  COMPLETA y no solo el ECE, porque dos curvas opuestas pueden dar el mismo ECE, y porque
  una curva sin los conteos por bin no se puede auditar.
- Que se exige ademas: que hay al menos un bin con al menos 20 unidades, porque con menos el
  ECE es ruido; y si no lo hay, se declara que el ECE de ese brazo no es interpretable.
- Tolerancia: 10 bins, enteros exactos en los conteos, banda muerta declarada 1e-9 para un
  bin vacio.
- Exito: la curva entera, los conteos y la linea de interpretabilidad.
- Fallo: falta un bin, o hay un NaN, o hay mas de la mitad de los bins vacios.
- STOP: no.

**A10. Cobertura y anchura, y el control degenerado de V03.**

- Que se verifica: que la cobertura del intervalo bayesiano al 90 por ciento esta medida y
  publicada; que la anchura media normalizada esta medida y publicada; que la metrica
  cobertura_menos_pesada_por_anchura esta medida y publicada; y que la cobertura del
  intervalo de V03 de anchura CERO, que es su exactitud, esta medida y publicada, y que el
  intervalo completo de anchura 1, [0, 1], tiene cobertura exactamente 1.0, lo que acorrla
  por arriba.
- Por que es STOP: sin el control de los dos extremos, cobertura y anchura no son
  interpretables. Un intervalo que cubre el cien por cien no dice nada, y uno estrecho que
  no cubre no aporta. Los dos controles convierten un par de numeros sin lectura en una
  afirmacion con lectura, y sin ellos la metrica de incertidumbre de la pregunta 1 seria
  exactamente el tipo de numero sin significado que A6 busca impedir.
- Tolerancia: la cobertura del intervalo completo tiene que ser 1.0 exacto, porque es
  aritmetica, y es la comprobacion de que el intervalo se construye bien. El resto se publica
  sin cota.
- Exito: las cinco cifras.
- Fallo: la cobertura del intervalo completo no es 1.0, o falta alguna de las cinco.
- STOP: si.

**A11. La regla de lectura de 1.5 se aplica y no se reinterpreta.**

- Que se verifica: que el informe contiene un campo lectura_pregunta_1 con el valor
  AFFIRMATIVO, NO_INFERIOR_SIN_AFIRMAR_SUPERIORIDAD, NEGATIVO o AMBIGUO_Y_POR_LO_TANTO_NADA, y
  que ese valor es el que sale de aplicar la regla de 1.5 a los dos intervalos, y que en el
  caso AMBIGUO el informe no hace ninguna afirmacion sobre la pregunta 1. Se exige tambien un
  campo ambiguedad_declarada con el valor NO_SON_INDEPENDIENTES, por la razon de que los
  cinco pliegues particionan las mismas 1483 filas y cada una se repite en cinco semillas.
- Tolerancia: el valor tiene que ser el que dicta la regla, sin reinterpretacion.
- Exito: el campo existe y coincide con la regla.
- Fallo: el valor no coincide con la regla, o el caso ambiguo publica una afirmacion. Esa
  ultima es la mas grave, porque es el modo de fallo de P78 y de P80, y en una pregunta que
  dice no competir por F1 un titular de F1 seria el colmo.
- STOP: si.

## 9. Bloque 2: la pregunta de complementariedad

**A12. Las cuatro medidas de complementariedad contra V03.**

- Las cuatro, sobre el OOF pareado y por etiqueta mas global: AUC de la probabilidad del
  miembro como predictor del error de V03; fraccion de acuerdo con V03; matriz de
  interseccion de aciertos y fallos; y diferencias por etiqueta.
- Tolerancia: seis decimales, enteros exactos para los conteos.
- Exito: las cuatro, por cada uno de los dos brazos.
- Fallo: falta alguna.
- STOP: no.

**A13. El rescate y el dano, sobre el OOF, con los tres alphas.**

- Que se mide: para los tres alphas de 3.5, 0.05, 0.10 y 0.15, con los umbrales de V03
  congelados, el corrected_cv del ensamblaje, su delta contra V03, el numero de pliegues a
  favor, el rescate, el dano, el neto, y el techo y el piso alcanzables, es decir cuantos
  cambios de decision serian necesarios para que el ensamblase igualara o superara a V03.
  Todo DESCRIPTIVO.
- Que se declara en el informe, con estas palabras: el rescate y el dano son un recuento de
  celdas y NO son comparables con el corrected_cv ni con ningun macro-F1. Y que el
  corrected_cv del ensamblaje sobre el OOF NO es una mejora de V03, porque el OOF ya se uso
  para elegir el alpha y para cualquier otra eleccion, luego es un numero optimista y no
  publicable como resultado.
- Se publica tambien la lectura con reoptimizacion del alpha sobre el OOF, marcada como
  DIAGNOSTICA_Y_SIN_LECTURA_INFERENCIAL, porque omitirla seria ocultar la lectura que mas
  favorece a la conclusion.
- Tolerancia: seis decimales para el corrected_cv, enteros exactos para los recuentos.
- Exito: las tres configuraciones completas y las dos declaraciones.
- Fallo: falta alguna configuracion, o falta una declaracion.
- STOP: no.

## 10. Bloque 3: no contaminacion, limites y sensibilidad

**A14. El calibrador se ajusta en internal, verificado por contadores y por mutacion.**

- Que se verifica: que el numero de llamadas a la funcion de ajuste del calibrador es igual
  al numero de combinaciones de pliegue y de calibrador interior, que es 5 por 5, y que
  CADA llamada usa filas del inner_calibration y NINGUNA fila del outer_validation; que la
  funcion que decide el alpha del ensamblaje tambien recibe solo filas internas; y que el
  numero de veces que se ajusta un parametro de calibracion sobre el OOF completo es 0, con
  contador explicito.
- Por que es STOP: es la regla que sostiene la honestidad de la pregunta 1. Si el calibrador
  se ajusta sobre el OOF y luego se mide la calibracion sobre el OOF, la curva de
  calibracion de la pregunta 1 es circular, y todo el bloque 1 no vale. P76 ya demostro que
  el camino anidado existe, luego el riesgo es que se tome el camino corto, no que no exista
  camino.
- Ademas, por mutacion, se comprueba que la comprobacion SABE FALLAR: se ejecuta una vez
  ajustando el calibrador sobre el OOF completo, y A14 tiene que detectarlo. Sin esa
  mutacion, A14 seria un chequeo que no puede fallar, y la bitacora de P81 dice que eso no
  prueba nada.
- Tolerancia: enteros exactos para los contadores; identidad de conjuntos de filas.
- Exito: los tres, mas la mutacion detectada.
- Fallo: cualquiera.
- STOP: si.

**A15. El limite de 5.5 esta verificado, no asumido.**

- Que se verifica: que el informe comprueba, en la propia corrida, si alguna libreria de
  inferencia bayesiana esta en el venv, y publica el resultado con la lista de nombres
  comprobados; y que el limite de 5.5 esta escrito, con la declaracion de si el anclaje
  externo quedo HECHO o quedo SIN HACER.
- Tolerancia: la lista de nombres comprobados esta publicada entera, y el valor del limit,
  HECHO o SIN_HACER, es el que sale de la comprobacion.
- Exito: los dos.
- Fallo: la comprobacion no se ejecuto, o el limite se escribio sin decir cual de los dos
  casos es.
- STOP: no. Es una medicion de alcance, y se declara con su propia naturaleza.

**A16. La rejeilla de alpha_0 y la sensibilidad de umbral, publicadas.**

- Que se publica: las tres magnitudes del numero de parametros efectivos, del numero de
  variables activas y del log-loss macro, para los TRES valores de alpha_0 de 5.2; y el
  log-loss macro y la dimension seleccionada para los TRES umbrales de la seleccion de 5.3.
  Todo por etiqueta, por pliegue y en el OOF completo.
- Tolerancia: seis decimales. No hay eleccion de la mejor combinacion, porque eso seria
  seleccion sobre el OOF, que es la contamination que A14 prohibe.
- Exito: las dieciocho cifras, o el numero exacto que la rejilla de las dos rejillas mas la
  agregada produzca, y publicacion de que ningun valor de la rejilla se elige como "el
  mejor".
- Fallo: falta alguna, o se publica un valor como elegido.
- STOP: no.

**A17. Dimension seleccionada por pliegue, y la salvedad de sesgo.**

- Que se publica: el numero de columnas que sobreviven a la seleccion, por pliegue y por
  umbral, con su minimo, su maximo y su media; y la comparacion de dimensiones con lo que la
  regresion L1 con C igual a 4.0 dejo fuera, que es un dato ya medido; y una declaracion
  escrita de que la seleccion por valor absoluto puede favorecer a las columnas largas, y que
  por eso un bayesiano que declara mas variables activas no ha ganado por ganar, sino que ha
  ganado por el criterio de seleccion.
- Que se exige ademas: que se comprueba que la dimension seleccionada es menor que el numero
  de filas del inner_train en cada pliegue, y si no lo es, que se dice explicitamente que la
  regresion no esta determinada.
- Tolerancia: enteros exactos. Banda muerta declarada: mayor o igual que el numero de filas
  del inner_train.
- Exito: las cifras y las dos declaraciones.
- Fallo: falta alguna, o hay un pliegue con dimension mayor o igual que filas y no se dice.
- STOP: no.

## 11. Bloque 4: redaccion y formulacion

**A18. Las cadenas de formulacion, de las dos preguntas y del proyecto.**

- Que se verifica: que el informe contiene, literales, estas cuatro cadenas:
  - V03 remains the best verified leaderboard result.
  - LA_PREGUNTA_UNO_NO_COMPITE_POR_F1_Y_NO_ACE_POR_EL_ORDENAMIENTO_POR_F1
  - LA_PREGUNTA_DOS_NO_BUSCA_SUPERAR_A_V03_SU_CRITERIO_ES_TENER_UN_PERFIL_DE_ERROR
    DISTINTO_Y_MEDIBLE
  - CON_1483_FILAS_Y_CELDAS_RARISIMAS_EL_PAPEL_ESPERADO_DEL_BAYESIANO_ES_MIEMBRO_DIVERSO
    NO_CAMPEON
- Y que NO aparece ninguna variante de la primera, ni ninguna frase que afirme que
  Word2Vec es mejor que V03, ni que el bayesiano lo supera.
- Tolerancia: coincidencia exacta, cero margen.
- Exito: las cuatro, y las tres ausencias.
- Fallo: cualquiera. STOP.
- STOP: si.

**A19. El F1 del miembro se publica como descriptiva, sin lectura inferencial.**

- Que se verifica: que el informe publica el f1_media_columnas del miembro bayesiano, con su
  desviacion entre semillas y su intervalo, con la etiqueta
  DESCRIPTIVA_Y_SIN_CUALQUIER_LECTURA_INFERENCIAL; y que ese numero NO aparece en ninguna de
  las dos reglas de lectura de las secciones 1 y 2; y que el informe dice, con esas palabras,
  que un F1 por debajo de V03 no refuta nada sobre la pregunta 1, porque la pregunta 1 no
  compite por F1, y que un F1 por debajo de V03 no refuta nada sobre la pregunta 2, porque la
  pregunta 2 no busca superar a V03.
- Por que es STOP: es la afirmacion que impide que el numero que sale con mas ruido termine
  siendo el titular. En P78 el numero con mas problemas era un bootstrap que excluia el cero,
  y el titular se escribio sobre el. En P83 hay un numero de F1 que va a salir, va a ser
  menor que el de V03, y va a ser el mas facil de citar y el menos util. A19 lo ata a su
  categoria antes de que exista.
- Tolerancia: presencia de la etiqueta y de las dos frases; ausencia del F1 en las reglas de
  lectura.
- Exito: las dos.
- Fallo: cualquiera.
- STOP: si.

## 12. Resumen de naturalezas

| Naturaleza | Cuantas | Ids |
|---|---|---|
| CONFIRMAR | 20 | A0 a A19 |
| REFUTAR | 0 | ninguna |
| NO APLICABLE POR CONSTRUCCION | 0 | ninguna |
| NO_EVALUABLE | 0 | ninguna |

Ninguna afirmacion es NO APLICABLE POR CONSTRUCCION y ninguna es NO_EVALUABLE, y las dos
cosas estan prohibidas como salida comoda: si una afirmacion no llega a ejecutarse por fallo
PROPIO del script, se marca ERROR con su tipo y su traza resumida, la corrida CONTINUA con
la siguiente, y ERROR no es un CONFIRMADO ni un REFUTADO ni un NO_EVALUABLE por el dato. Lo
unico que aborta de verdad es un STOP.

## 13. Criterio de parada

Se para, con SystemExit, sin registrar ningun otro numero, si:

- A0 falla: cambia el hash de un artefacto congelado. STOP INMEDIATO y aviso a Carlos.
- A1 falla: se ha leido test o el holdout, o se ha gastado una evaluacion del holdout.
- A2 falla: se ha reentrenado V03 o se ha reoptimizado un umbral.
- A3 falla: el OOF de referencia no es el canonico, luego la pregunta 2 se queda sin contra.
- A4 falla: la seleccion de variables se ha hecho fuera del inner_train, luego hay
  contamination en los dos bloques.
- A5 falla: la posterior de M1 se ha descrito como exacta, luego el bloque 1 no describe lo
  que se ejecuto.
- A6 falla: el numero de parametros efectivos no se mueve con alpha_0, luego no es el
  numero que dice ser.
- A7 falla: la cuenta cerrada de M2 no se reproduce, luego no se puede confiar en la
  aritmetica del brazo de referencia.
- A10 falla: el control del intervalo completo no da cobertura 1.0, luego el intervalo no se
  construye bien.
- A11 falla: el caso ambiguo publica una afirmacion.
- A14 falla: el calibrador se ha ajustado sobre el OOF, luego la pregunta 1 es circular.
- A18 falla: falta una cadena de formulacion o aparece una variante.
- A19 falla: el F1 se ha usado como criterio.
- Aparece un numero sospechosamente perfecto, por ejemplo un log-loss menor que 0.05, o una
  cobertura de intervalo igual a 1.000000. Se investiga antes de reportar nada.
- Se viola cualquier frontera de 13.5 o de 2.

Criterio de parada de presupuesto: 25 minutos de reloj. P83 es un modelo lineal sobre una
matriz rala de 1483 filas, luego el presupuesto es de minutos, no de horas, y esa asimetria
con P82 se declara para que no se lea como que P83 es el experimento importante. No lo es: es
el barato.

## 14. Lo que P83 no hace

- No genera submission, no consulta el leaderboard, no lee test, no lee el holdout ni su
  fichero de identificadores, y no gasta ninguna de las cinco evaluaciones del holdout.
- No reentrena, no modifica, no sobrescribe el artefacto Word2Vec 100d. De hecho P83 no lo
  usa: trabaja sobre la TF-IDF de V03.
- No reentrena V03, no toca sus umbrales congelados 0.44, 0.51 y 0.46.
- No modifica ni borra ningun script de P71 a P82, ni sus informes, ni sus JSON.
- No afirma que el bayesiano supere a V03, ni que improves la clasificacion, ni que su
  posterior sea exacta.
- No afirma que Word2Vec sea mejor que V03.
- No usa el OOF para ajustar el calibrador, ni para elegir el alpha, ni para seleccionar
  variables.
- No elige umbral, tolerancia, margen ni criterio despues de ver un numero.
- No ancla su posterior contra una libreria de inferencia bayesiana, salvo que A15
  compruebe que hay una en el venv y la use.

## 15. Formulacion

Toda conclusion que P83 escriba lleva esta frase, y solo esta frase sobre el estado del
proyecto:

    V03 remains the best verified leaderboard result.

No se afirma jamas que Word2Vec sea mejor que V03. No se afirma jamas que el modelo
bayesiano supere a V03. Ni siquiera si el ensamblaje sube el corrected_cv sobre el OOF, y
el motivo esta escrito en 3.5: ese numero es optimista porque el OOF ya se uso para elegir
lo que se luego combina.

    V03 remains the best verified leaderboard result.
