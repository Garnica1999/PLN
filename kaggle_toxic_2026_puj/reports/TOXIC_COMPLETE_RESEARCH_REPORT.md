# TOXIC - Informe completo de la investigacion acumulada

Generado por `experiments/p60_12_total.py`. Fecha 2026-09-28T18:06:45+00:00. Commit 5a33594.

Todo lo que sigue esta DERIVADO de fuentes estructuradas y no transcrito: el estado del
campeon de `CHAMPION.md`, el historial de `experiments.csv`, y los artefactos de fase.
Donde un dato no existe en su fuente se escribe NOT RECORDED en vez de rellenarlo.

## 1. Estado del campeon

```text
V03 corrected_cv = 0.7772114525165721
V03 LB           = 0.78306
V03 sha256       = 7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
submissions registradas en CHAMPION.md = 22
fecha de CHAMPION.md = 2026-09-28
```

> V03 remains the best verified leaderboard result.

## 2. Alcance de la investigacion en una linea

El registro canonico tiene **336 experimentos** repartidos en **68 familias**, con **0 filas sin decision legible**. La clasificacion automatica de la columna decision, con las reglas declaradas en el propio script, da:

| clase de decision | n |
|---|---:|
| OTRA | 181 |
| CERRADA | 88 |
| ADOPTADA | 56 |
| INVALIDADA | 5 |
| MARGINAL | 4 |
| PROMETEDORA | 2 |

La cifra merece una advertencia de honestidad: la clasificacion es por palabras clave
sobre el texto de la decision, y el proyecto escribe las decisiones en prosa y en dos
idiomas a lo largo de 60 fases. Un reparto como este NO es un recuento denn y no debe
leerse como tal; sirve para localizar donde esta el peso de la investigacion.

## 3. Familias, por peso

| familia | n | adoptada | prometedora | marginal | cerrada | invalidada | otra |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2 | 58 | 30 | 0 | 0 | 21 | 0 | 7 |
| EBD | 26 | 3 | 0 | 0 | 6 | 0 | 17 |
| V8 | 14 | 0 | 0 | 0 | 10 | 0 | 4 |
| V4 | 14 | 1 | 0 | 0 | 10 | 0 | 3 |
| V10 | 12 | 0 | 0 | 0 | 10 | 0 | 2 |
| V6 | 10 | 0 | 0 | 0 | 10 | 0 | 0 |
| FIN | 10 | 5 | 0 | 0 | 0 | 0 | 5 |
| PRE | 10 | 0 | 0 | 0 | 0 | 0 | 10 |
| V11 | 9 | 0 | 0 | 0 | 3 | 0 | 6 |
| MOD | 9 | 5 | 0 | 0 | 0 | 0 | 4 |
| V32 | 8 | 0 | 0 | 0 | 0 | 0 | 8 |
| P60 | 8 | 0 | 0 | 1 | 3 | 0 | 4 |
| V12 | 6 | 0 | 0 | 0 | 6 | 0 | 0 |
| V31 | 6 | 0 | 0 | 0 | 0 | 0 | 6 |
| V29 | 6 | 0 | 0 | 0 | 0 | 0 | 6 |
| REP | 6 | 2 | 0 | 0 | 0 | 0 | 4 |
| V33 | 5 | 0 | 0 | 0 | 0 | 0 | 5 |
| V9 | 5 | 2 | 0 | 0 | 0 | 0 | 3 |
| P59 | 5 | 0 | 0 | 3 | 0 | 0 | 2 |
| E | 5 | 0 | 0 | 0 | 0 | 0 | 5 |
| HIB | 4 | 0 | 0 | 0 | 0 | 0 | 4 |
| V28 | 4 | 0 | 0 | 0 | 1 | 0 | 3 |
| DL | 4 | 0 | 0 | 0 | 0 | 0 | 4 |
| V7 | 4 | 0 | 1 | 0 | 3 | 0 | 0 |
| ENS | 4 | 0 | 0 | 0 | 0 | 0 | 4 |
| V2 | 4 | 0 | 0 | 0 | 0 | 0 | 4 |
| V44 | 4 | 0 | 0 | 0 | 0 | 0 | 4 |
| V35 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| V40 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| V45 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| V36 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| V37 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| V38 | 3 | 0 | 0 | 0 | 0 | 1 | 2 |
| V39 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| V30 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| V43 | 3 | 0 | 0 | 0 | 0 | 1 | 2 |
| V26 | 3 | 0 | 0 | 0 | 0 | 1 | 2 |
| AUD | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| V15 | 3 | 1 | 0 | 0 | 0 | 0 | 2 |
| V13 | 3 | 1 | 0 | 0 | 1 | 0 | 1 |
| V25 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| V41 | 2 | 0 | 0 | 0 | 1 | 1 | 0 |
| V51 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| V50 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| V47 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| V46 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| V42 | 2 | 0 | 0 | 0 | 0 | 1 | 1 |
| V14 | 2 | 0 | 0 | 0 | 2 | 0 | 0 |
| V34 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| S05 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| S07 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| B5 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| E00 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| HOLD | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| V56 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| B1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| TH | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| S08 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| OPT | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| S04 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| P8 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| PA | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| S06 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| V27 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| R2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| S02 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| S03 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| R1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |

## 4. Las tres familias que el proyecto mantuvo vivas

### 4.1 V03, la referencia que nunca se movio

V03 es un modelo lineal sobre la union de TF-IDF word(1,2) y char_wb(3,5), con
`class_weight=balanced`, `C = 2,0`, solver liblinear y umbrales 0,44 / 0,51 / 0,46. Su
`corrected_cv` canonico es exactamente el que se cita al principio de este informe, y su
sha256 se ha verificado al inicio y al cierre de cada uno de los ocho scripts de P60 y
en el script de cierre. Ninguna fase posterior lo ha modificado.

La razon de que sea tan dificil de superar esta medida en el registro: `V28-01`
midio que los errores de HS, TR y AG se solapan con phi de +0,42 a +0,51, luego las tres
etiquetas fallan sobre los mismos textos. Cualquier metodo que actue sobre una sola
etiqueta choca contra ese techo estructural, y por eso las familias que sirogationaron
familias que si cambiaron el eje, la geometria y el objetivo, son las unicas con una
razon real de mejorar.

### 4.2 P59, la frontera de senal dentro de las familias conocidas

P59 pregunto si quedaba alguna fuente de senal dentro de las familias ya conocidas, con
cinco experimentos consumidos de doce disponibles:

```text
baseline                                     {'corrected_cv': '0.7772114525165721', 'LB': 0.78306, 'sha256': '7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e'}
candidatos                                   0
conclusion                                   P59 no encontro una fuente de informacion nueva que mejore V03. La dificultad se detecta y no se corrige; la frontera se mueve y las decisiones no; la calibracion mueve la metrica sin corregir nada
experimentos_consumidos                      5
familia                                      P59_FINAL
fecha                                        2026-09-28T14:38:20+00:00
finalistas                                   0
formulacion                                  V03 remains the best verified leaderboard result.
pistas                                       [{'track': 'P59-A', 'estado': 'CLOSED', 'experimentos': 1, 'corrected_cv': 0.7796406441, 'delta_vs_V03': 0.0024291916, 'folds': '3/5', 'neto': -18}, {'track': 'P59-C', 'estado': 'MARGINAL', 'experimentos': 0, 'corrected_cv': None, 'delta_vs_V03': None, 'folds': 'n/a', 'neto': None}, {'track': 'P59-D', 'estado': 'CLOSED', 'experimentos': 0, 'corrected_cv': None, 'delta_vs_V03': None, 'folds': 'n/a', 'neto': None}, {'track': 'P59-G', 'estado': 'CLOSED_BY_DUPLICATION', 'experimentos': 0, 'corrected_cv': None, 'delta_vs_V03': None, 'folds': 'n/a', 'neto': None}, {'track': 'P59-H', 'estado': 'CLOSED_BY_FUNCTIONAL_EQUIVALENCE', 'experimentos': 0, 'corrected_cv': None, 'delta_vs_V03': None, 'folds': 'n/a', 'neto': None}, {'track': 'P59-I', 'estado': 'MARGINAL', 'experimentos': 1, 'corrected_cv': 0.7785171129, 'delta_vs_V03': 0.0013056604, 'folds': '4/5', 'neto': 4}, {'track': 'P59-B', 'estado': 'CLOSED', 'experimentos': 1, 'corrected_cv': 0.7765868373, 'delta_vs_V03': -0.0006246152, 'folds': '3/5', 'neto': -1}, {'track': 'P59-T', 'estado': 'CLOSED_BY_DUPLICATION', 'experimentos': 0, 'corrected_cv': None, 'delta_vs_V03': None, 'folds': 'n/a', 'neto': None}, {'track': 'P59-U', 'estado': 'MARGINAL', 'experimentos': 1, 'corrected_cv': 0.77869146, 'delta_vs_V03': 0.0014800075, 'folds': '2/5', 'neto': 6}, {'track': 'P59-E', 'estado': 'MARGINAL', 'experimentos': 1, 'corrected_cv': 0.7773636094, 'delta_vs_V03': 9.88776e-05, 'folds': '2/5', 'neto': -5}, {'track': 'P59-F', 'estado': 'CLOSED_BY_DUPLICATION', 'experimentos': 0, 'corrected_cv': None, 'delta_vs_V03': None, 'folds': 'n/a', 'neto': None}]
presupuesto                                  {'principales': 12, 'reserva': 2, 'usados': 5}
submissions                                  0
```

La conclusion de P59, en una frase, es que **la calibracion mueve la metrica sin mover la
correccion**: en P59-B el sesgo de la agregacion produjo +0,003367, que supera el gate,
con 11 rescates contra 15 danos. Quien solo mirase la metria concluiria que hay
mejora donde no hay nada. Y en esa fase el mejor `corrected_cv`, el mejor `delta` y la
mejor consistencia de pliegues pertenecieron a tres pistas distintas, y ninguna era
candidata.

### 4.3 P60, mecanismos nuevos con apoyo de literatura

P60 evaluo doce familias de la literatura sobre aprendizaje multi-etiqueta, consumio
**8 experimentos** y cerro **4 pistas sin gastar ninguno**. Cero submissions, cero consultas al leaderboard.

| pista | corrected_cv | delta del mecanismo | pliegues | rescate | dano | neto | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 0.624338862 | -0.152926 | 0/5 | 334 | 1552 | -1218 | **CLOSED** |
| C | 0.7772647319 | +0.000000 | 0/5 | 2 | 1 | 1 | **CLOSED** |
| D | 0.7768328831 | -0.000432 | 0/5 | 3 | 3 | 0 | **CLOSED** |
| G | 0.4497522457 | -0.294907 | 0/5 | 278 | 868 | -590 | **CLOSED** |
| H | 0.4686780062 | -0.072307 | 0/5 | 391 | 2655 | -2264 | **CLOSED** |
| I | 0.4761602463 | -0.301104 | 0/5 | 318 | 890 | -572 | **CLOSED** |
| J | 0.7755546945 | -0.001710 | 2/5 | 25 | 23 | 2 | **CLOSED** |
| L | 0.777600986 | +0.000336 | 2/5 | 2 | 1 | 1 | **MARGINAL** |

**Ninguna pista produjo un candidato.** El gate exige delta de al menos +0,003 y 4 de 5
pliegues, y ninguna lo alcanzo. Ningun bootstrap se ejecuto, porque el bootstrap solo se
ejecuta tras superar el gate, y es preferible no tener ninguno a tener el intervalo de
una mejora que no existe.

## 5. Las cuatro pistas de P60 cerradas sin consumir experimento

| pista | referencia | veredicto |
|---|---|---|
| B | W-ASL | CERRADA POR DUPLICACION FUNCIONAL |
| E | LOGIT-ADJ/LDAM | CERRADA POR DUPLICACION FUNCIONAL |
| F | LDL | CERRADA POR DUPLICACION FUNCIONAL |
| K | CONFORMAL | CERRADA POR DUPLICACION FUNCIONAL |

Los cuatro cierres son la parte mas barata de la fase, y la mas informativa sobre el
estado del problema:

* **E, logit adjustment**: sobre un modelo lineal, ajustar el logit es desplazar el
  intercepto, y con umbral fijo un desplazamiento de intercepto ES un cambio de umbral,
  tanto post-hoc como dentro de la perdida. Cierre de raiz, con prueba algebraica, y no
  con un experimento.
* **B, weighted ASL**: se descompone en ponderacion por frecuencia de etiqueta, que es
  `class_weight=balanced`, y suavizado de coocurrencia, que es la familia de dependencia
  entre etiquetas que V4-E01-E03 cerro en doce variantes. Dos patas, ninguna nueva.
* **F, label distribution learning**: con tres etiquetas anidadas el subespacio factible
  tiene cuatro puntos, medido en este corpus, y es exactamente el que V4-E ya enumero.
* **K, conformal prediction**: convertir un conjunto conformista en una decision binaria
  exige cortar la puntuacion de no conformidad en un punto, que es un ajuste de umbral.

## 6. Los hallazgos que sobreviven a P60

### 6.1 La estructura de etiquetas es anidada, y eso es informacion nueva

```text
prevalencia_HS                   0.43695212407282535
prevalencia_TR                   0.3000674308833446
prevalencia_AG                   0.38233310856372216
P_TR1_dado_HS0                   0.0
P_AG1_dado_HS0                   0.026347305389221556
P_AG1_dado_TR1                   0.946067415730337
negativos_TR_no_definidos        1.0
negativos_AG_no_definidos        0.9736526946107784
```

El 100 por ciento de los negativos que ve TR en entrenamiento son etiquetas **no
definidas**, y el 97 por ciento en AG. Eso es ruido dependiente de la etiqueta, medido.

### 6.2 Ese ruido es localizable, y en un lugar concreto

C1 y C2 construyeron seis senales por par y las compararon contra el control con el
control-conditioned. El resultado localiza la senal:

| etiqueta | AUC del control | mejor senal | AUC incremental | ganancia |
|---|---:|---|---:|---:|
| HS | 0,9917 | rango | 0,5547 | +0,0252 |
| TR | **1,0000** | centroide | 0,5338 | +0,0288 |
| AG | 0,9961 | estructural | 0,6627 | **+0,1129** |

El control de TR tiene AUC **exactamente 1,0000**: la confianza out-of-fold ordena
perfectamente todos los errores de TR, luego no hay margen residual para ninguna segunda
senal. Y dentro de la region sin definir de AG el AUC del control es **0,1814**, muy por
debajo de 0,5, es decir la confianza esta ANTICORRELACIONADA con el error ahi. Eso es lo
que hace que la senal estructural tenga ganancia, y es informacion genuina.

### 6.3 Pero no es accionable, y el eje queda acotado entre dos extremos medidos

La intervencion suave, con un factor de fiabilidad DERIVADO de las tasas de error de
training, fue un no-op exacto: delta 0,0000000000 y cero celdas cambiadas, porque el
factor salio entre 0,993 y 1,009. La version dura, con peso cero, ya la habia cerrado
`V14-00` con -0,116581. El eje queda acotado entre dos extremos medidos, factor cero es
catastrofico y factor uno no hace nada, y el hueco intermedio se declara en vez de
explorarse con una plaza de reserva para perseguir un numero mejor.

### 6.4 Un gap check que se midio a si mismo

La pista de transporte optimo exigia demostrar que su mecanismo no era una familia
cerrada. Con objeto de Dirac, W1 es la media de distancias a las muestras de la clase,
que es la distancia al centroide, luego OT **colapsa de forma demostrable** a la
familia cerrada. La version con objeto distribucional no. Y esa distincion se midio con
un gap check AUTOCALIBRADO, que compara la diferencia Dirac contra distribucion con el
sesgo de la propia metrica. El sesgo es de 0,016 por ciento y la diferencia de 43,559,
o sea **2842 veces** el ruido. La condicion de novedad se cumple y esta medida.

Y aun asi la operacion nueva es peor: -0,072307 contra su propio control de centroide.

### 6.5 Mecanismos que se activaron y no ayudaron, frente a los que no se activaron

Esta distincion es la que mas informo de P60, y conviene no mezclarla:

* **No se activaron**: el corte duro de la perdida asimetrica, con 0,00 por ciento de
  negativos con perdida exactamente cero, porque un modelo bien ajustado nunca lleva la
  probabilidad de una etiqueta ausente por debajo de 0,05. Lo que si actuo fue la
  potencia del negativo, un factor 15 de reescalado, que es un ponderado de clase, y esa
  familia la cerro `V8-07`.
* **Se activo y fue demasiado debil**: el riesgo minimax por grupos, que sobrepondera al
  grupo con mas riesgo en un 2,47 por ciento y mueve 3 celdas de 4449, y ademas empeora el
  grupo que pretendia proteger.
* **Se activo y fue fuerte, y aun asi danio**: el meta-reweighting, con el 20,77 por ciento
  de pesos negativos y una razon entre percentiles 90 y 10 de 5,35, que degrada la
  decision en -0,301104. El ponderado no es degenerado; lo que falta es un conjunto
  retenido limpio, que es la hipotesis del metodo y aqui no se cumple.
* **No se activo por construccion**: el reescalado suave de la region sin definir, con un
  factor derivado que quedo en 0,993 a 1,009.

## 7. Errores de instrumentacion encontrados en P60

Se registran porque un informe que solo lista los resultados no permite entender por que
algunos numero cambian entre dos ejecuciones del mismo script:

1. **SupCon devolvia NaN.** El log-softmax con la diagonal enmascarada a menos infinito
   multiplicado por una mascara que vale cero en la diagonal da menos infinito por cero.
   La diagonal se anula DESPUES de la resta del log-particion.
2. **Gradiente de la perdida asimetrica.** En gamma mas igual a cero la derivada del
   termino positivo es menos g, es decir menos uno menos p, y no menos p. Ademas el valor
   no multiplicaba por la constante de regularizacion mientras el gradiente si, lo que
   daba un gradiente numerico exactamente la mitad del analitico.
3. **El objetivo de riesgo por grupos media regularizacion, no agregacion.** Cada riesgo
   de grupo es una media, luego la perdida queda en escala O(1) frente a la escala O(n)
   del objetivo de V03, y el brazo media un C efectivo unas 1186 veces mayor. Daba
   -0,163. Corregido con un factor constante n, que deja el gradiente exacto, el delta
   real es +0,000336.
4. **Un parametro que se ignoraba en silencio.** El optimizador compartido aceptaba un
   objetivo propio y no lo usaba, luego una pista midio un delta de exactamente cero en
   quince ajustes, que es un no-op y no un resultado. El resultado real fue -0,001710.
   Esa pista lleva ahora un guardia que detiene la ejecucion si las dos soluciones no
   difieren.
5. **La metrica de transporte era mas ruidosa que la diferencia medida.** Con 128 puntos
   de cuantil el sesgo era del 44 por ciento, mayor que la diferencia de 43,6 que se
   pretendia medir. El gap check autocalibrado concluyo correctamente que no se podia
   afirmar nada, y el fallo era de la metrica y no del metodo.
6. **Un detector con sobreajuste.** El score compuesto de la pista C se ajustaba y se
   evaluaba sobre las mismas filas: 0,5407 de AUC donde la version out-of-fold da 0,5070,
   luego el 0,7093 de ganancia incremental era artefacto.
7. **Referencia de exactitud mal construida.** El coste de Dirac se comparaba contra la
   media NO PONDERADA de distancias en vez de la ponderada por la medida de la clase.

## 8. Cumplimiento de restricciones, verificado y no declarado

```text
transformers = 0        BERT = 0            RoBERTa = 0         LLM = 0
embeddings preentrenados = 0                FastText = 0        Word2Vec = 0
GloVe = 0                                   sentence-transformers = 0
clasificadores externos = 0                  etiquetas externas = 0
corpus toxicos externos = 0                  hate speech externo = 0
pseudo-etiquetas = 0                         test labels = 0     test training = 0
test calibration = 0                         test feature selection = 0
holdout reuse = 0                            leaderboard tuning = 0
submissions creadas en P60 = 0              consultas a Kaggle en P60 = 0
```

La verificacion de esto la hace `experiments/p60_14_auditoria.py`, que lee el codigo de
los scripts de P60 y los artefactos, y no se limita a escribir ceros.

## 9. La pregunta final, respondida en sus partes

> Despues de P59 y P60, existe evidencia de que algun mecanismo nuevo aporte informacion o
> capacidad realmente complementaria a V03?

**No, y conviene separar por que.**

* **Deteccion**: si hay informacion nueva. La region sin definir de AG tiene la confianza
  anticorrelacionada con el error y un indicador estructural que aporta +0,1129 de AUC
  residual. Es real, esta medida y no se conocia antes.
* **Calibracion**: no. La unica mejora grande de P59 era un sesgo de agregacion, y P59-U
  quedo marginal. La calibracion es el unico hilo vivo y es una hipotesis de umbral que
  V8-09 cerro de raiz.
* **Representacion**: hay dos mecanismos genuinamente nuevos, el contraste por solapamiento
  y el transporte optimo, y los dos verificados como nuevos. Ninguno ayudo.
* **Optimizacion**: cinco mecanismos probados, y la leccion es que en este corpus el
  ponderado por clase esta agotado, la forma de la perdida esta agotada y la
  identificabilidad del meta-reweighting falla por falta de etiquetas limpias.
* **Informacion nueva**: existe, es pequena y es local, y no tiene via de accion rentable.
  El eje esta acotado entre factor cero, que es catastrofico, y factor uno, que no hace
  nada.

La conclusion que sostiene el conjunto: **V03 no esta limitado por la falta de un
mecanismo de aprendizaje mejor, sino por el techo estructural de la correlacion de
errores entre las tres etiquetas.** Los dos unicos ejes que podrian esquivarlo, la
geometria aprendida y el objetivo sobre grupos, se ejecutaron en P60 y ninguno produjo
mejora. El gasto futuro mas rentable no es una pista mas de estas familias, sino
atacar ese techo, y eso ya se ha intentado en V4-H01, V28-01 y ENS-01 a ENS-04 con tres
sondas al leaderboard que quedaron por debajo de V03.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_12_total.py`. Sin push.*

<!-- P61 2026-09-28 -->
## P61 (2026-09-28) - Capacidad, diversidad y distribucion de probabilidad

Tres experimentos: `P61-01` LightGBM con Optuna anidado, `P61-02` ensemble con pertenencia
y pesos anidados, `P61-03` segunda particion con control negativo. Ademas la
reconciliacion de los nueve problemas de instrumentacion de P60, la EDA de las
probabilidades OOF de V03 y el inventario con diversidad de los veinte canales del pool de
P50.

```text
P61-01  LightGBM + Optuna anidado    0.7513651502
         familia -0.0302397191
         busqueda +0.0043934168
         total -0.0258463023   2/5   CLOSED
P61-02  ensemble anidado             0.7835159596  +0.0063045070  gate superado
         control de atribucion: el modelo nuevo SUBTRAIA -0,000850
P61-03  segunda particion            test 1 -0.0011527969
         test 2 holdback medio -0.0015908404 con 7/10
         test 3 control negativo +0.0059889238   REJECTED
```

**El hallazgo que mas importa.** El control negativo NO colapsa. Con los tres peores
canales del pool, entre 0,04 y 0,24 por debajo de V03, la ganancia del ensemble sigue
siendo +0.0059889238. Un promediado ponderado que contiene a V03 mejora el OOF casi con
seguridad porque los pesos se eligen para exploitear su residuo sobre las mismas filas, y
el bootstrap no lo detecta porque mide ruido de muestreo y no sesgo de seleccion. Esto
explica de una sola causa el fracaso historico de ENS-01 a ENS-04.

**El diagnostico que queda.** Entre el 60 y el 68 por ciento de los errores de V03 son
errores de alta confianza, con la distancia al umbral POR ENCIMA de la de los aciertos.
El limite de V03 no es de decision ni de calibracion: es de representacion.

```text
no evidence of exploitable remaining model capacity/diversity
V03 remains the best verified leaderboard result.
```
<!-- /P61 2026-09-28 -->

<!-- P62 2026-09-29 -->
## P62 (2026-09-29) - Representacion, estructura linguistica y error de alta confianza

Cinco experimentos, tres pistas cerradas sin gastar nada, y una correccion de lectura.

```text
P62-01  L palabra__POS     -0,009572  1/5  CLOSED
P62-02  M triple lexico   -0,011236  0/5  CLOSED
P62-03  curva de presupuesto   ningun K mejora
P62-04  POS contra lemas   ambos pierden 8/8, el POS ayuda 6/8
P62-05  cobertura min_df=1  -0.0033508936  CLOSED
```

**La interaccion existe y es medible.** La synergy palabra-POS es +0.0657, el
17.41 por ciento de la senal de la palabra. El problema no
es que falte informacion, es que no cabe: repaga el
17.4 por ciento del coste de anadir el canal.

**La cobertura tampoco es la via.** `min_df=2` conserva solo el
28.3 por ciento de las formas superficiales
presentes, y `min_df=1` lo sube al 99.5 por
ciento y resta.

**La correccion.** Yo lei mal mi propia EDA: el perfil de longitud se llamaba
`estructura_corta` pero su cuartil superior es lo largo, luego la delta negativa decia que
V03 es MEJOR en los largos, no peor en los cortos. La comprobacion independiente lo
desmiente: V03 alcanza 0.9639 de AUC en los cortos y su F1
es 0,94 contra 0,72 en los largos. El limite esta en el ORDEN de los textos largos, donde
V28-01 ya habia localizado el techo estructural.

```text
no evidence of exploitable remaining model capacity in the linguistic representation
V03 remains the best verified leaderboard result.
```
<!-- /P62 2026-09-29 -->

<!-- P63 2026-09-29 -->
## P63 (2026-09-29) - Representacion a nivel de registro, estructura del dataset y composicion

Seis experimentos, dos auditorias y un dataset auxiliar. Ninguno supera a V03, y lo
interesante no son los negativos sino que cada uno lleva un mecanismo MEDIDO.

```text
P63-00  ID forensics      D. STRUCTURED_IDENTIFIER, confianza alta, 0 experimentos
P63-02  id predictivo     0,419136 contra 0,424038 de un control de igual dimension
P63-03  relacional        el bloque esta 92,92 por ciento vacio
P63-04  composicion       brecha especifica +0,0083528268, y +0,1055 en solitario
P63-05  texto largo       la brecha en MUY_LARGA es -0,009437
P63-06  integracion       brecha +0,0085891400, MARGINAL
P63-07  residuo           los bloques restan al margen, luego son redundantes con el
                         margen y no con los n-gramas
```

**El id no es un contador.** V15-00 lo afirmo sin medirlo. Train y test son dos muestras
disjuntas del mismo bloque contiguo, con interseccion 0 y union
2885 sobre 4999 posiciones. Es un indice de
coleccion, y su contenido es la posicion en un ensamblado, que no correlaciona con la
etiqueta.

**La composicion en ratios porta senal y no cabe.** Es un resultado distinto del de P62, no
uno contrario: un canal disperso de varios cientos de columnas compite por el presupuesto
L2 de un espacio de 21000 dimensiones, y 42 proporciones acotadas no compiten. Y aun asi
no mejora a V03, porque lo que aportan ya esta implicito en la confianza de V03.

**La hipotesis mas antigua del repositorio queda refutada.** V30-01 escribio que el efecto de
la URL y la mencion cambia de signo con la longitud, y nunca gasto un experimento. El bloque
de posiciones es PEOR que un bloque aleatorio del mismo ancho precisamente en la region
muy larga, con -0.009437. Refutada en su prediccion, no solo
no confirmada.

**El limite que queda es de orden, no de decision.** El margen hasta el techo moviendo solo
el corte va de +0.0000 en la region corta a +0.0890 en la
muy larga, y en la region CORTA es exactamente +0.0000: el umbral de V03 ya
es el optimo ahi sobre las tres etiquetas.

**Dos correcciones cambiaron la direccion de una conclusion** y tienen fila propia en el
registro. El confound de escala de P63-04 paso el delta de -0,1899987838 a
-0,0102687019 y cambio el signo de la brecha. El techo de P63-06 paso el margen de MUY_LARGA
de +0,1589 a +0,0890.

```text
no evidence of exploitable remaining model capacity in the record-level and
aggregate-composition dimensions of this representation

V03 remains the best verified leaderboard result.
```
<!-- /P63 2026-09-29 -->

<!-- P64 2026-09-29 -->
## P64 (2026-09-29) - Representacion del tweet largo y distribucion de la evidencia interna

Cuatro experimentos, dos auditorias, trece figuras. Ninguno supera a V03, y la respuesta
tiene dos mitades que no se pueden separar.

**La estructura interna es REAL.** P59-07 habia explicado su fracaso por cobertura, no por
concepto: con ventana de 32 tokens solo el
9.6 por ciento de las filas tiene mas
de una unidad, mientras que la segmentacion por puntuacion llega al
50.6 por ciento. Un factor de
5.29 en las celdas alcanzables, que no es una variacion
de parametro.

**Y el leave-one-out, que P59 nunca computo, muestra el mecanismo.** El 80,88 por ciento de
los documentos multi segmento de la region MUY_LARGA tienen segmentos que empujan la decision
en signos opuestos, la entropia interna sube de 0,1048 a 0,4565 con la longitud, y la
concentracion cae de 0,7554 a 0,6170. En los tweets largos la decision no descansa en un
segmento dominante sino en un reparto contradictorio. Eso es exactamente lo que una bolsa de
n-gramas aplana.

**Y no aporta.** El bloque de evidencia queda -0.0140959650 frente al
margen de V03, y el leave-one-out -0.0004650203, que es casi nula.
Tampoco gana en la region larga, que es donde P63 midio el margen, y el margen tampoco mejora
ahi, luego el problema de MUY_LARGA no es de representacion ni de confianza.

**El perfil segmental es longitud disfrazada.** El mejor rasgo da
0.589847 de AUC frente a
0.600506 de contar tokens, luego queda
cerrado como rasgo.

**La razon de fondo.** El margen de V03 solo ya explica sus propios errores con AUC 0,770875
contra el acierto, y el mejor rasgo de evidencia interna llega a 0,605302. La estructura no
falta: esta dentro de la confianza de V03.

```text
the internal evidence distribution of long tweets is real, measured and reproducible,
and V03 already carries it inside its own confidence

V03 remains the best verified leaderboard result.
```
<!-- /P64 2026-09-29 -->

<!-- P65 2026-09-29 -->
## P65 (2026-09-29) - Representacion de largo alcance, alcance de operadores y jerarquia

Cuatro experimentos y una auditoria, trece figuras. Ninguno supera a V03, y la respuesta tiene
una forma que no estaba prevista.

**LA RESPUESTA: la senal existe y la adicion la rechaza.** Cinco candidatos de este proyecto
baten a los dos controles, el de capacidad y el de la confianza de V03, y ninguno mejora:

| candidato | contra el control de anchura | contra el margen de V03 | delta vs V03 |
|---|---:|---:|---:|
| P64-07 alcance de negacion | +0,0201566146 | +0,0088975231 | -0,0021826774 |
| P65-01 word (1,3) | +0.0596671770 | +0.0071281317 | -0,0039520688 |
| P65-03 cohesion documental | +0.0090835614 | +0.0083711943 | -0,0027090062 |

Lo que falta no es informacion: es sitio. Unas cientos de columnas compitiendo por el
presupuesto L2 de un espacio de veinte mil, y lo que la columna nueva necesita decir es mas
pequeno que lo que cuesta decirlo. Eso es una afirmacion sobre la ARQUITECTURA de la adicion,
y es falsable: predice que la misma evidencia fundida en el espacio de probabilidad, y no
concatenada en el de caracteristicas, sobrevive.

**EL GAP ANALYSIS CERRO MAS DE LO QUE ABRIO.** 6 abiertas y
10 cerradas, leidas en el codigo y no en el registro. Los skip-grams ya los
habia medido V6-05, con un vocabulario un 83 por ciento mayor y un resultado neutro. La
negacion y la cita las habian medido E2-07 y E2-08, y despues P64-06 y P64-07.

**EL CONTROL DE IMPLEMENTACION REPRODUCE A V6-05.** El rango de hueco 1 a 2, que es
exactamente el suyo, da
-0.0010515908 frente al -0,0001 que el
registro. Luego la implementacion de los pares con hueco es la misma, y los rangos que V6-05 no
alcanzo, de tres a cinco y de seis a diez tokens, tampoco aportan.

**Y LA LONGITUD DEL TWEET NO ES EL MECANISMO.** Un especialista ajustado en LARGA y MUY_LARGA
es BIT A BIT el mismo numero que un reentrenado sobre las mismas
714 filas sin ningun criterio de longitud:
+0.0000000000. Un negativo asi no es *faltan datos*,
que seria una excusa valida: dice que la especializacion no es el mecanismo, punto. Y el
especialista gana
+0.0212 en MUY_LARGA y
pierde -0.0257 en LARGA, que tiene
el triple de filas, luego hay una mejora local real que la metrica primaria rechaza.

```text
the information about long-range relations is present, measurable and not what V03 already
knows, and none of it fits through concatenation

the length of a tweet is not the mechanism either: a model retrained on the long rows with no
length criterion is bit for bit the specialist

V03 remains the best verified leaderboard result.
```
<!-- /P65 2026-09-29 -->

<!-- P66 2026-09-29 -->
## P66 (2026-09-29) - Late fusion: la evidencia de largo alcance en el espacio de scores

Dos experimentos y dos auditorias, trece figuras. Y es la primera vez que la senal de largo
alcance mueve el numero.

**LA RESPUESTA A LA PREGUNTA DE LA FASE: SI, Y ES ATRIBUIBLE AL CANDIDATO.**

| fusion | delta vs V03 | pliegues | brecha de atribucion |
|---|---:|---:|---:|
| MI_arbol + residual anidada | +0.0026743167 | 4/5 | +0.0041133588 |
| MI_arbol + residual con score barajado | -0.0014390421 | | |

La brecha de atribucion es la cifra que manda: el mismo mecanismo con el score del candidato
**barajado** pierde casi todo.

**Y MI_arbol es el unico que funciona porque es NO LINEAL.** Las correlaciones entre z_V03 y
z_C de los siete candidatos van de +0,9294 a +0,9988: son casi la misma puntuacion. El AUC del
residuo deja a cinco de siete por debajo de 0,5, luego su parte unica destruye informacion en
vez de anadirla. El de MI_arbol es el unico por encima del azar, y su modelo es el unico no
lineal de la lista. Si el error de V03 fuera una funcion lineal del error de MI_arbol, la
regresion residual no dejaria nada.

**EL MATIZ, Y VA PEGADO AL RESULTADO.** De las
53 celdas que la fusion rescata, solo
6 son errores de
alta confianza y las demas estan pegadas al umbral, luego es sobre todo **recalibracion**. Y el
bootstrap por 1415 grupos da IC95
[-0.0044311634, +0.0102108745], que
**incluye el cero**. Veredicto PROMISING, no FINALIST.

**EL INVENTARIO FUE EL PROBLEMA REAL.** Ningun script de P64 y P65 escribio sus
probabilidades a disco, y `features/` esta en `.gitignore`. Los scores se regeneraron y se
verificaron contra el numero de cada fase: 5 de 7 reproducen con diferencia de 1e-11, uno es
no verificable y uno tiene una diferencia declarada. El regenerador **se paro** cuando dos no
cuadraron.

**Y UNA CORRECCION QUE CAMBIO EL VEREDICTO DEL MEJOR RESULTADO.** `blend_prob` usaba siempre
la probabilidad REAL del candidato y nunca la barajada, luego su control de barajado salia
identico al real POR CONSTRUCCION, la brecha daba cero falso, y el mejor resultado de la fase
queda INVALIDATED por un motivo que no era el suyo. Corregido, ese mismo resultado pasa a
+0,0033 con brecha +0,0093.

```text
adding columns to a twenty thousand dimension space does not fit, and correcting a three
column score does, for the SAME evidence and with the SAME base model

the only measured exception is MI_arbol, and its distinguishing feature is that the model
producing it is NON LINEAR

and the caveat: it is mostly recalibration, because 6 of 53 rescues are high confidence and
the IC95 includes zero

V03 remains the best verified leaderboard result.
```
<!-- /P66 2026-09-29 -->

<!-- P67 2026-09-29 -->
## P67 (2026-09-29): el residuo de MI_arbol es MIXED, no una senal transferible

- Categoria del mecanismo: **MIXED_CON_ATRIBUCION_PARCIAL**.
- Reentreno anidado de un escalar por etiqueta: `-0.0029731513`, es decir la calibracion **no** explica el efecto.
- AUC del residuo de MI_arbol contra la etiqueta **fuera** de la banda: `0.579597`, sobre 0,5. Los otros seis candidatos caen por debajo.
- Sustitucion cruzada: `4 de 7` candidatos dan positivo con la misma operacion.
- Enriquecimiento de los rescates: `1.338302`.
- Efecto en celdas: `4.01` de 1483 filas.
- 1 de 12 tracks principales ejecutados, 0 de 4 reservas.
- 0 nuevas representaciones, 0 submissions, 0 Kaggle, 0 holdout, 0 test.
- Figuras: 6 en `figuras/p67/`.

> V03 remains the best verified leaderboard result.
<!-- /P67 2026-09-29 -->
\n<!-- POST-P67 2026-09-29 -->\n## Post-P67 (2026-09-29): el gap analysis y una linea nueva cerrada

- Poblacion **B = 215 celdas** donde V03 falla y alguien acierta. El 28,4 por ciento de sus
  errores. **122 exclusivas** de un solo candidato.
- Once familias clasificadas: 6 `CLOSED`, 2 `INVALIDATED`, 1 `DUPLICATE`, 1 `COMPLEMENTARY`,
  1 `PARTIALLY_OPEN` (MI_arbol).
- `P68-H1` (enrutado por estructura del ejemplo): **INVALIDATED**, delta
  -0.0018992302, 1 de 5 pliegues.
- El techo era `+0,0390871336`. El enrutador no lo captura porque la estructura predice
  dificultad (AUC 0,6080), no ganador (AUC 0,5491).
- 1 experimento principal, 2 controles, 1 ablacion. Sin grid.

> V03 remains the best verified leaderboard result.
<!-- /POST-P67 2026-09-29 -->\n