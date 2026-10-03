# P59 - Informe final de investigacion

Generado por `experiments/p59_12_informe_final.py`. Fecha 2026-09-28T14:38:20+00:00.
Todos los numeros proceden de los artefactos JSON de cada pista, leidos en tiempo de
generacion. Este documento no contiene cifras propias.

## Baseline, congelado y verificado

```text
V03 corrected_cv  0.7772114525165721
V03 LB             0,78306
V03 sha256         7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
```

## Resumen de las once pistas

| # | pista | familia | estado | exp | corrected_cv | delta | folds | bootstrap | resc | dano | neto |
|---:|---|---|---|---:|---|---|---|---|---:|---:|---:|
| 1 | P59-A | geometria local | CLOSED | 1 | 0.7796406441 | 0.0024291916 | 3/5 | [-0.008991, 0.000674] | 55 | 73 | -18 |
| 2 | P59-C | ruido de etiqueta formal | MARGINAL | 0 | n/a (no se entreno) | n/a (no se entreno) | n/a | n/a (diagnostico) | n/a | n/a | n/a |
| 3 | P59-D | influencia | CLOSED | 0 | n/a (no se entreno) | n/a (no se entreno) | n/a | n/a (diagnostico) | n/a | n/a | n/a |
| 4 | P59-G | seleccion por estabilidad | CLOSED_BY_DUPLICATION | 0 | n/a (no se entreno) | n/a (no se entreno) | n/a | n/a | n/a | n/a | n/a |
| 5 | P59-H | estructura no negativa | CLOSED_BY_FUNCTIONAL_EQUIVALENCE | 0 | n/a (no se entreno) | n/a (no se entreno) | n/a | n/a | n/a | n/a | n/a |
| 6 | P59-I | perdida robusta | MARGINAL | 1 | 0.7785171129 | 0.0013056604 | 4/5 | NO ejecutado (gate no superado) | 8 | 4 | 4 |
| 7 | P59-B | evidencia local | CLOSED | 1 | 0.7765868373 | -0.0006246152 | 3/5 | NO ejecutado (gate no superado) | 39 | 40 | -1 |
| 8 | P59-T | calibracion del score local | CLOSED_BY_DUPLICATION | 0 | n/a (no se entreno) | n/a (no se entreno) | n/a | n/a | n/a | n/a | n/a |
| 9 | P59-U | fusion global + local | MARGINAL | 1 | 0.7786914600 | 0.0014800075 | 2/5 | NO ejecutado (gate no superado) | 17 | 11 | 6 |
| 10 | P59-E | regularizacion por grafo | MARGINAL | 1 | 0.7773636094 | 0.0000988776 | 2/5 | NO ejecutado (gate no superado) | 35 | 40 | -5 |
| 11 | P59-F | n-gramas generativas | CLOSED_BY_DUPLICATION | 0 | n/a (no se entreno) | n/a (no se entreno) | n/a | n/a | n/a | n/a | n/a |

Experimentos consumidos: **5 de 12 principales**, 2 de reserva intactos. Cerradas sin consumir experimento: G, H, T y F. Ninguna pista fue candidata ni finalista.

## Una pista por una, en orden de ejecucion

### 1. P59-A - geometria local

**Estado CLOSED | 1 experimento(s) | dependencia: ninguna**

Motivacion: la geometria local predice los errores de V03 con AUC 0,725. Novedad: parcialmente duplicada, y el precedente V6-07 ya habia probado lo mismo con otra senal. Resultado: corrected_cv 0.7796406441, delta 0.0024291916, 3/5 pliegues, IC95 [-0.008991, 0.000674], rescate 55 dano 73 neto -18. Decision: CLOSED. Predice el error con AUC 0,725 y aun asi no aporta nada accionable

### 2. P59-C - ruido de etiqueta formal

**Estado MARGINAL | 0 experimento(s) | dependencia: ninguna**

Motivacion: estimar el ruido de etiqueta de forma formal para ver si autoriza downweighting. Novedad: nueva, y se ejecuto como diagnostico sin entrenar. Resultado: s_resid alcanza AUC 0,9730, pero el control 1 - p_true alcanza 0,9741, luego el estimador **no supera al control** y el criterio de downweighting no se cumple. Ademas 36 de las 42 celdas no rescatables son error de alta confianza, que es justo lo que downweighting no arregla. Decision: MARGINAL.

### 3. P59-D - influencia

**Estado CLOSED | 0 experimento(s) | dependencia: ninguna**

Motivacion: medir la influencia de cada ejemplo de entrenamiento sobre la prediccion. Novedad: nueva. Resultado: el rango de influencia reproduce el LOO real con Spearman 0,9993 y concordancia de signo 1,0000, luego el metodo es correcto. Pero la influencia se asocia al error con rho +0,3955 y a la confianza con rho -0,5345, una razon de 0,740: es **peor medida de dificultad que la propia confianza de V03**. Decision: CLOSED. Diagnostico, no experimento, luego sin corrected_cv.

### 4. P59-G - seleccion por estabilidad

**Estado CLOSED_BY_DUPLICATION | 0 experimento(s) | dependencia: ninguna**

Motivacion: seleccionar por estabilidad en lugar de por metrica. Novedad: duplicada. El precedente V8-03 ya probo seleccion por submuestras a tres umbrales, con -0,003308, -0,004628 y -0,018890, en 1 de 5, 1 de 5 y 0 de 5 pliegues, y su propia nota dice que no se reintenta con mas agresividad. Decision: CLOSED por duplicacion, sin consumir experimento.

### 5. P59-H - estructura no negativa

**Estado CLOSED_BY_FUNCTIONAL_EQUIVALENCE | 0 experimento(s) | dependencia: ninguna**

Motivacion: buscar estructura latente no negativa compartida que comprima sin perder complementariedad. Novedad: documentalmente nueva, con 0 filas que mencionen NMF. Pero la forma funcional no aporta: NMF(8) reconstruye **peor** que SVD truncada del mismo rango, 0,093047 frente a 0,091040, una brecha de -0,002007, con esparsidad 0,5357 frente a 0,0238 y 8 de 8 componentes con top-5 distinto. Es legible y no es informacion nueva. Decision: CLOSED por equivalencia funcional, sin entrenar.

### 6. P59-I - perdida robusta

**Estado MARGINAL | 1 experimento(s) | dependencia: ninguna**

Motivacion: la perdida logistica es demasiado sensible a residuos extremos. Novedad: nueva; 0 filas mencionan una perdida redescendiente y V34-01 lista margin/robust loss con profundidad (0) UNEXPLORED. Configuracion: biweight de Tukey, una sola funcion, tau 4,685 constante estandar, sigma 1 porque el residuo de devianza ya es estandarizado, y una normalizacion que hace que la perdida robusta coincida con el log-loss hasta segundo orden para que la regularizacion efectiva no cambie.

Control de tres brazos: A es V03 y reproduce el canonico bit a bit; B es mi propio ajuste sobre la logistica pura y clona a A con 3 celdas de 4449. El gradiente analitico se verifico contra diferencias centrales con error relativo 1e-09, y con sigma tending a infinito el brazo robusto recupera el brazo logistico con coseno 1,00000000, que es la prueba de consistencia del objetivo.

Resultado: corrected_cv 0.7785171129, delta 0.0013056604, 4/5 pliegues, las tres etiquetas suben, rescate 8 dano 4 neto 4, y solo cambian 12 de 4449 celdas. **El mecanismo no es el hipotetizado**: el redescendiente nunca activa, con 0 cambios entre los errores de alta confianza; la influencia **sube** con ratio 1.011161 y Spearman 0.99924 entre las dos distribuciones; y los residuos empeoran, percentil 99 de 1.276045 a 1.347712. Decision: MARGINAL. El delta pequeno no puede atribuirse al mecanismo redescendiente.

### 7. P59-B - evidencia local

**Estado CLOSED | 1 experimento(s) | dependencia: ninguna**

Motivacion: cambiar la unidad de representacion de documento a ventana local, porque V03 puede diluir evidencia toxica concentrada. Novedad: `B_NOVELTY = True`, `FUNCTIONAL_EQUIVALENCE = FALSE`. El precedente V4-D01-D04 probo ventanas locales como tokens, pero alla la ventana es un canal extra apilado al documento y la unidad de puntuacion sigue siendo el tweet; aqui la ventana ES la unidad de puntuacion. DL-02 es el precedente del principio de agregacion, campos locales con max, y perdio en familia densa.

Factibilidad registrada ANTES de mirar validation: los tweets son cortos, mediana de 15 tokens y maximo 51, luego con ventana 32 solo el 8.23 por ciento tiene mas de una ventana. El cambio solo puede actuar en 366 de 4449 celdas, y eso es una cota previa, no una promesa.

Resultado: corrected_cv 0.7765868373, delta -0.0006246152, 3/5 pliegues, rescate 39 dano 40 neto -1, con acierto 0.4937 en las celdas que cambian, es decir una moneda al aire. Y la descomposicion es lo relevante: en el estrato multi-ventana, donde el mecanismo si actua, el corrected_cv mejora +0,003367 pero el recuento es rescate 11 contra damage 15, neto -4. La mejora no viene de corregir errores sino del sesgo al alza del max, +0,0549 de media, al desplazar celdas a traves de un umbral fijo. Fuera de ese estrato hay 53 cambios con neto +3 en tweets que por construccion no fueron segmentados, que son efecto del conjunto de entrenamiento y no del mecanismo local. Decision: CLOSED.

### 8. P59-T - calibracion del score local

**Estado CLOSED_BY_DUPLICATION | 0 experimento(s) | dependencia: B1**

Motivacion: la evidencia local puede ser util pero su agregacion produce una escala de probabilidad distinta de la de V03, y una calibracion podria separar el problema de representacion del problema de decision. Novedad: **existe equivalencia funcional**, y por eso no se ejecuto. Dos patas.

Pata algebrica: Platt es una transformacion afin en el logit, luego con umbral fijo t, p_cal >= t equivale EXACTAMENTE a p_raw >= sigmoid((logit(t) - b)/a), que es reseleccionar el umbral por etiqueta. Una reparametrizacion monotona no aporta informacion. Y el propio encargo excluye el threshold sweep al definir T.

Pata de precedente: V8-09 midio sobre quince sondas independientes que la F1 frente a la tasa de positivos predicha es unimodal y que **V03 esta exactamente en su maximo**, lo que cierra de raiz la familia entera de calibraciones, umbrales y pesos por etiqueta. Y V10-04 probo Platt, isotonica y beta de tres parametros sobre los scores de V03 con el calibrador anidado: la calibracion funciona de verdad, ECE de 0,058293 a 0,024801, pero el delta medio anidado de Platt es +0,000148 con IC95 [-0,005221, +0,005932], y su decision cierra con clausula pre-registrada de no reintentar mas variantes. Decision: CLOSED_BY_DUPLICATION, 0 experimentos.

### 9. P59-U - fusion global + local

**Estado MARGINAL | 1 experimento(s) | dependencia: B1**

Motivacion: aunque el score local aislado sea peor, puede contener evidencia complementaria, y una fusion continua conserva la representacion global. Novedad: base nueva, mecanismo ya explorado. ENS-01 a ENS-04 combinan cuatro bases globales, V4-H01 y S08 usan un meta de 36 rasgos, y FIN-08/09/10 enviaron candidatos de meta al leaderboard y los tres quedaron por debajo de V03. Lo nuevo es que **ninguna fusion previa uso un score calculado sobre una unidad de representacion distinta**. Por que esta no cae en el cierre de T: la fusion cambia el ranking siempre que los dos scores discrepen, y una reparametrizacion monotona no puede hacer eso.

Configuracion: p = (1 - alpha) p_V03 + alpha p_local, con UN alpha compartido por las tres etiquetas, elegido por UNA regla, maximizar el macro-F1 a los umbrales congelados sobre las predicciones inner-OOF del training fold exterior. La rejilla tiene 54 valores fijos y **el nulo alpha = 0 esta dentro**, luego la regla puede volver siempre a V03.

Resultado: corrected_cv 0.7786914600, delta 0.0014800075, 2/5 pliegues, rescate 17 dano 11 neto 6, acierto 0.6071 en las celdas que cambian. Los alphas por pliegue fueron [0.3, 0.45, 0.55, 0.0, 0.75]. **El dato decisivo**: el alpha oracle, el que habria maximizado el score externo, es 0.65 y aporta +0.0006441507, **menos** que la seleccion anidada y honesta. La mejor fusion con un alpha unico en todo el corpus apenas mueve la metrica, luego el delta de la regla anidada viene de adaptar alpha pliegue a pliegue y no de senal transferible. En multi-ventana da rescate 3 y damage 0, el unico punto de todo el bloque donde el score local produce correcciones puras, y son 3 celdas de 4449. Decision: MARGINAL.

### 10. P59-E - regularizacion por grafo

**Estado MARGINAL | 1 experimento(s) | dependencia: ninguna**

Motivacion: la estructura de vecindad en el espacio de V03 puede usarse para **regularizar el entrenamiento**, no para predecir localmente. Novedad: nueva. Cero filas del registro mencionan regularizacion por grafo, manifold regularization, un termino cuadratico sobre un grafo de documentos, PageRand, random walk o redes neuronales sobre grafo. El precedente mas cercano, V4-F01-F04, usa la vecindad para **propagar etiquetas**, que es predecir, y cerro con techo nulo. V4-A01, V4-A02, E2-20-ESPECIALISTAS y V6-07 usan vecindad para predecir o ponderar. Ninguno regulariza.

Y conviene no esconder el previo nulo: V4-F01-F04 midio que un vecino discrepante aporta solo +0,018 de error con IC95 [-0,013, +0,049], no significativo. Un grafo puede regularizar sin que los vecinos compartan error, luego no cierra la pista, pero se ejecuta sabiendo que el diagnostico previo salio nulo.

Configuracion: cosmNN con k = 10 sobre el TF-IDF de V03, filas de entrenamiento, simetrizado y reescalado a norma espectral unitaria de X'WX; objetivo 0,5 w'w + (lambda/2) w'X'WX w + C sum omega perdida(s), con lambda = 0,1 unico y sin rejilla. El grafo SOLO regulariza: nunca vota, nunca recupera. Control: el brazo 0 es mi propio optimizador sin grafo y clona a V03 con 3 celdas de 4449, luego la diferencia entre brazos es del grafo.

Resultado: corrected_cv 0.7773636094 con el grafo, frente a 0.7772647319 sin el. Efecto puro del grafo 0.0000988776, unas treinta veces mas pequeno que la dispersion entre pliegues. 2/5 pliegues, rescate 35 dano 40 neto -5, acierto 0.4667 en las 75 celdas que cambian: una moneda al aire. Decision: MARGINAL, indistinguible del ruido, y no se convierte en voto kNN para rescatarlo.

### 11. P59-F - n-gramas generativas

**Estado CLOSED_BY_DUPLICATION | 0 experimento(s) | dependencia: ninguna**

Motivacion: un modelo generativo de n-gramas puede exploiting evidencia de secuencia que un discriminativo sparse no representa igual. Novedad: **existe equivalencia funcional**, y por eso no se ejecuto. Tres patas.

Pata algebrica: sobre una representacion de n-gramas fija, todo modelo generativo condicional de clase colapsa a log P(c|x) = log P(c) + sum_f theta_{f,c}, que es la forma de naive Bayes y de log-odds. El unico parametro libre es el estimador de theta, y todo estimador de theta es una estrategia de suavizado.

Pata de familia: MOD-02 probo ComplementNB sobre la union TF-IDF de V03 y dio -0,069949 con 0 de 5 pliegues. E2-15 probo el ratio log-count NB-SVM y dio -0,0234 con 0 de 5 pliegues, ganancia de ensemble -0,0057 y correlacion de residuales con V03 de 0,9243. V37-01 probo BM25 y dio -0,0027860149 con neto -6 y correlacion 0,9874, con su propia frase de que BM25 no aporta una fuente de senal nueva. V26-03 probo Tsetlin Machine y fue REJECTED. Cero filas describen un modelo generativo como tal.

Pata que cierra la especificacion exacta: el unico grado de libertad de este modelo es el suavizado, y F esta especificado justo como un orden y una estrategia de suavizado. V8-04 midio ese parametro: el ratio por etiqueta sobre la TF-IDF de V03 da -0,058496 con 1e-6 y +0,004745 con 1e-3, un salto de +0,063241 entre dos valores que solo difieren en esa constante, y su lectura es que el suavizado sostiene el metodo entero. Su decision pre-registra por escrito que no se barre el suavizado. F es un punto suelto de un barrido cerrado de antemano. Decision: CLOSED_BY_DUPLICATION, 0 experimentos.

## Conclusion global, separando cinco efectos que la fase muestra distintos

La pregunta es si P59 encontro una fuente de informacion nueva que mejore V03. La respuesta
requiere no confundir cinco cosas que la fase ha separado:

### 1. Efecto diagnostico: REAL y fuerte, pero no accionable

La dificultad **se detecta**. La geometria local de V03 predice sus errores con AUC 0,725194 con IC95 [0,7061, 0,7446]. El estimador formal de consistencia latente alcanza AUC 0,973019. La funcion de influencia reproduce el leave-one-out real con Spearman 0,9993 y concordancia de signo 1,0000. Tres mecanismos independientes, todoscorrectos. Lo que no existe es una traduccion de esa deteccion a una correccion.

### 2. Efecto calibracion: mueve la CIFRA sin mover la CORRECTITUD

Es el hallazgo mas importante de la fase y hay que decirlo sin suavizarlo. La calibracion de V03 funciona: ECE medio de 0,058293 a 0,024801 con Platt, y su delta en la metrica es +0,000148 con IC95 que incluye el cero. En P59-B el sesgo al alza del max, +0,0549 de media, produjo la mayor mejora aparente de toda la fase, +0,003367 en el estrato alcanzable, **superando el propio umbral de gate de +0,003**, con un recuento de celdas de rescate 11 contra damage 15. Una mejora de F1 sin una sola correccion neta. Quien solo mirase la metrica habriann concluding que hay una mejora donde no hay nada.

### 3. Efecto representacion: mueve la FRONTERA sin mover las DECISIONES

La perdida robusta cambia los coeficientes, con cambio relativo de norma 0.045852 de media y coseno 0.999295, es decir la frontera se mueve de verdad. Y aun asi solo cambian 12 de 4449 celdas y ninguna de ellas esta entre los 268 errores de alta confianza. La representacion por ventanas, por su parte, solo puede actuar en 366 celdas por construccion, porque la mediana del corpus es de 15 tokens.

### 4. Efecto decision:POSITIVO pequeno e INCONSISTENTE

Las tres pistas que movieron el F1 al alza lo hicieron por menos del gate y sin consistencia: P59-I con 0.0013056604 en 4 de 5 pliegues, P59-U con 0.0014800075 en 2 de 5, y P59-E con 0.0000988776 en 2 de 5. Ninguno paso el gate, luego **a ninguno se le ejecuto bootstrap**, y el unico IC95 que existe en toda la fase para una mejora, el de P59-A, es [-0.008991, +0.000674] e **incluye el cero**.

### 5. Informacion complementaria: NO se produjo en ninguna pista

Este es el veredicto sobre la pregunta de fondo. Los tres recuentos que miden si un mecanismo corrige lo que V03 no puede corregir:

```text
P59-A  rescate 55  dano 73  neto -18   con IC95 que incluye el cero
P59-I  rescate 8  dano 4  neto 4   y 0 correcciones entre errores de alta confianza
P59-B  rescate 39  dano 40  neto -1   y dentro del estrato alcanzable 11 contra 15, neto -4
P59-U  rescate 17  dano 11  neto 6   y el alpha oracle solo da +0.0006441507
P59-E  rescate 35  dano 40  neto -5   con acierto 0.4667 en las celdas que cambian
```

Ninguna de las cinco produce informacion complementaria transferible. Tres tienen neto negativo, y las dos positivas son de tres y cuatro celdas.

### Respuesta a la pregunta

**P59 no encontro una fuente de informacion nueva que mejore V03.** La evidencia consolidada de once pistas es que los mecanismos capaces de detectar dificultad producen diagnosticos reales y no accionables; los que modifican marginalmente la frontera mueven coeficientes y no decisiones; los que alteran la representacion local no llegan a cambiar las celdas que importan; y la unica palanca que mueve la metrica de forma apreciable, la calibracion, la mueve sin corregir nada.

Conviene decirlo con su alcance exacto: esto **no** es una prueba de que V03 sea optimo. Es una afirmacion sobre el conjunto de mecanismos que esta fase pudo probar con el presupuesto y el protocolo disponibles.

## Comparacion final

```text
V03 corrected_cv   0.7772114525
V03 LB             0,78306  (unico resultado verificado en el externo)

mejor corrected_cv de P59   0.7796406441  (P59-A)
mejor delta de P59          +0.0024291916  (P59-A, 3/5 pliegues)
mejor consistencia          4/5  (P59-I)

ningun candidato supero el gate de +0,003 con 4 de 5 pliegues
ningun bootstrap se ejecuto, porque el gate previo no se supero en ninguna pista
0 submissions, 0 consultas a Kaggle, 0 uso de holdout, 0 uso de test
```

| candidato | corrected_cv | delta | folds | bootstrap | resc | dano | neto |
|---|---|---|---|---|---:|---:|---:|
| P59-A | 0.7796406441 | 0.0024291916 | 3/5 | [-0.008991, 0.000674] | 55 | 73 | -18 |
| P59-I | 0.7785171129 | 0.0013056604 | 4/5 | NO ejecutado (gate no superado) | 8 | 4 | 4 |
| P59-B | 0.7765868373 | -0.0006246152 | 3/5 | NO ejecutado (gate no superado) | 39 | 40 | -1 |
| P59-U | 0.7786914600 | 0.0014800075 | 2/5 | NO ejecutado (gate no superado) | 17 | 11 | 6 |
| P59-E | 0.7773636094 | 0.0000988776 | 2/5 | NO ejecutado (gate no superado) | 35 | 40 | -5 |

Ninguno de estos tres es candidato: los tres fallan el gate de delta y ninguno se bootstrapio. Y ningun leaderboard se afirmo mejorado, porque ninguno se envio.

## Formulacion

> V03 remains the best verified leaderboard result.

## Cumplimiento de la fase completa

```text
0 transformers | 0 BERT | 0 RoBERTa | 0 LLM | 0 embeddings
0 clasificadores externos | 0 etiquetas externas | 0 pseudo-etiquetas
0 test labels | 0 test training | 0 holdout reuse | 0 leaderboard tuning
0 submissions | 0 consultas a Kaggle
V03 sha256 verificado e intacto en el cierre de cada pista
OOF canonico reproduce 0,7772114525165721 exacto en el cierre de cada pista
1415 grupos canonicos, 0 partidos entre pliegues, verificado en cada pista
```

*Generado por `experiments/p59_12_informe_final.py`. Sin push.*