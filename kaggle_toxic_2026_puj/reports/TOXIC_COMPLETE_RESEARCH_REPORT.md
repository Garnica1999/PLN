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
