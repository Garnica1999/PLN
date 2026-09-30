# P78: control anidado Word2Vec frente a azar, con la normalizacion alineada

Addendum al informe `reports/p78_control_anidado.json`. Es la redaccion del resultado con sus
limites. Los numeros no se reescriben: se leen con su incertidumbre y con lo que no prueban.

Pre-registro: `docs/p78_preregistro.md`, sha256 `6737f43522c1c9dd0fc45d474d04ab92a7a230aa6b674ee83645588db20c9a0b`,
commiteado en `dabf07d` **antes** de existir el script.

## 1. Que se pregunto y que se respondio

Si el embedding Word2Vec aporta senal frente a una matriz aleatoria pareada, con early stopping
estrictamente anidado dentro de `outer_train`, y con el texto de entrada normalizado igual que
el corpus. Cuatro brazos, tres semillas pareadas, cinco pliegues agrupados.

```text
A  Word2Vec 100d   entrada CRUDA   macro-F1 media 0.592444   sd 0.016930
B  aleatoria pareada  CRUDA         macro-F1 media 0.569137   sd 0.006724
C  Word2Vec 100d   entrada NORMALIZADA   macro-F1 media 0.650999   sd 0.017442
D  aleatoria pareada  NORMALIZADA       macro-F1 media 0.626106   sd 0.005729

P1 = C - D   +0.024893   sd 0.021588   5/5 pliegues   IC95 bootstrap (0.007902, 0.040645)
P2 = A - B   +0.023307   sd 0.023590   4/5 pliegues   IC95 bootstrap (0.006199, 0.039429)
P3 = C - A   +0.058555   sd 0.015017   5/5 pliegues   IC95 bootstrap (0.040710, 0.076529)
P4 = D - B   +0.056969   sd 0.001272   5/5 pliegues   IC95 bootstrap (0.040363, 0.073252)
```

Criterio pre-registado, seccion 7, aplicado en orden: **rama 1**. El gate `EPS_GATE_DELTA = 0.003`
se cumplio.

## 2. El resultado que sostiene el gate, y por que es fragile

La rama 1 se cumple como regla contable. La lectura estadistica es mas debil de lo que el gate
sugiere, por tres motivos que hay que enunciar juntos.

**El intervalo que decide no mide lo que la rama presume.** El bootstrap pareado por clusters
remuestrea tweets y promedia sobre las tres redes ya entrenadas. Es incertidumbre de **muestra**
condicionada a esas tres redes: propaga **cero** variabilidad de reentrenamiento. El estimador
puntual de P1 esta a 2.0 desviaciones de semilla del cero, y la desviacion entre semillas es
0.021588, mayor que el semiancho del intervalo, 0.016372. El IC t sobre las tres semillas, con n=3
y df=2, es [-0.028736, +0.078522] y **contiene el cero**.

**Los deltas por semilla son heterogeneos:**

```text
P1 = C - D    s20261130 +0.030964   s20261131 +0.042796   s20261132 +0.000919
P2 = A - B    s20261130 +0.012511   s20261131 +0.050364   s20261132 +0.007048
P3 = C - A    s20261130 +0.075886   s20261131 +0.050377   s20261132 +0.049401
P4 = D - B    s20261130 +0.057433   s20261131 +0.057944   s20261132 +0.055530
```

P1 pasa de +0.0428 a +0.0009 segun la semilla. El gate de +0.003 se cumple en dos de tres y
**casi no se cumple en la tercera**. Con 1483 filas de dev y este dataset, tres semillas no
distinguen el efecto del embedding del ruido de reentrenamiento.

**"5/5 pliegues" no son cinco observaciones independientes.** Los cinco pliegues particionan las
mismas 1483 filas y cada una se repite en tres semillas: quince unidades altamente correlacionadas
sobre un unico conjunto. El recuento de signos no es replicacion.

Se puede afirmar: el efecto es positivo en las quince unidades pareadas, la estimacion puntual es
+0.025 y el bootstrap por clusters excluye el cero. **No** se puede afirmar que el efecto se
reproduzca al reentrenar.

## 3. El hallazgo limpio: el efecto de la normalizacion

P4 mide el efecto de alinear la normalizacion en el brazo **aleatorio**, es decir sin embedding
que lo sostenga. Es +0.056969 con desviacion entre semillas de **0.001272**, trece veces menor que
la del brazo, y con las tres semillas en +0.0574, +0.0579 y +0.0555. Un bootstrap y una t sobre
semillas dan la misma lectura, luego el veredicto es invariante al intervalo que se elija.

Descomposicion del total, en la semilla primaria:

```text
C - B  total           +0.088397
  (C - D)  embedding   +0.030964    35.0 por ciento
  (D - B)  normalizacion +0.057433  65.0 por ciento
```

**El 65 por ciento de la ganancia total la produce una transformacion determinista de cadenas, y
no el embedding.** Ese es el resultado de primer orden del experimento, y es mas grande que el que
buscabamos medir.

El desglose del OOV lo explica. La cobertura sube de 68.64 a 88.73 por ciento y el OOV cae de
31.36 a 11.27:

```text
casing     5787 ocurrencias, 100 por ciento OOV,  68.3 por ciento de todo el OOV,  4465 rescatadas
mencion     449                                  5.3 por ciento,                   449 rescatadas
numero      187                                  2.2 por ciento,                   187 rescatadas
hashtag      93, 52 OOV                          0.6 por ciento,                     0 rescatadas
url        1493, solo 3 OOV                      0.0 por ciento,                     0 rescatadas
sin motivo  1989 OOV                            23.5 por ciento,                     0 rescatadas
total       5101 de 8467 OOV rescatados, 60.25 por ciento
```

y las **filas totalmente muertas**, las que con cero tokens cubiertos dejan las tres ramas de
filtros a cero, caen de **70 a 1**. Ese es el mecanismo: 69 filas que el normalizador rescata.

## 4. Lo que NO se puede decir sobre la interaccion

`P1 - P2 = P3 - P4 = +0.001586` identicamente, semilla a semilla. Es el termino de interaccion, y
es pequeno frente a los dos efectos principales.

Lo que se puede decir: **aditividad aproximada en la media**. La normalizacion aporta lo mismo con
y sin embedding.

Lo que **no** se puede decir: que el efecto sea ortogonal al embedding. No hay IC95 del termino de
interaccion, y la desviacion de P3, 0.015017, es doce veces la de P4, 0.001272. Esa asimetria es en
si misma una interaccion de varianza, y un efecto puramente de entrada no deberia depender de la
semilla en un brazo y no en el otro. Que la media de tres valores dispersos de a 0.0016 es
**cancelacion**, no un efecto pequeno.

## 5. La razon por la que P73 y P74 estaban sobreestimados

P73 reporto +0.0316259 y P74 +0.02427 con diez de diez semillas positivas. Con protocolo limpio
y el tokenizado tal cual estaba, P2 = **+0.023307**. Dos efectos explican la diferencia: el
checkpoint se elegia con el pliegue externo, y el 68 por ciento de su OOV era ruido de caja.

**Reetiquetado: P73 y P74 pasan de "evidencia reproducible" a "exploratorios, protocolo
contaminado".** Sus numeros se conservan intactos.

## 6. Seccion de corpus: cerrada

El pre-registro la condicionaba a que persista el OOV y a que la procedencia lo permita.

La cobertura ya esta en 88.73 por ciento **sin tocar el corpus**. Del OOV residual de la variante
normalizada, 11.27 por ciento, solo los 650 tokens lexicamente ajenos serian recuperables con mas
corpus, un 2.5 por ciento de los tokens. El techo de esa fase es de 88.73 a cerca de 91.3 por
ciento de cobertura. **La cobertura dejo de ser el cuello de botella.**

La procedencia tampoco lo permite. La fuente es tweets de COVID-19 en Espana de abril de 2020; el
conjunto de desarrollo es texto de foro sobre registro civil e instituciones. Mas del mismo
genero no aporta vocabulario de ese dominio.

Desglose de los 1989 tokens OOV sin motivo asignado, medido sobre el corpus real: 898, el 45.1 por
ciento, son forma ortografica recuperable, y la causa de fondo no es el corpus sino que el
propio conjunto de desarrollo llega sin tildes; 441, el 22.2 por ciento, estan en el corpus con
frecuencia 1 o 2 y los descarto `MIN_COUNT = 3`; 650, el 32.7 por ciento, son de dominio.

Lo que si procede no es mas corpus sino `MIN_COUNT` de 3 a 2, que no requiere corpus nuevo y
recupera 441 tokens, un 1.7 por ciento. No justifica una fase por si solo.

El margen grande esta en la calidad del vector, y ahi ataca P79.

## 7. El control pareado, verificado por fuera

El especialista de NLP senalo una brecha: el script no prueba que la matriz aleatoria tenga las
normas de fila de la matriz Word2Vec. Se comprueba por fuera:

```text
las tres semillas: norma de la fila aleatoria igual a la de la fila Word2Vec, diferencia maxima 4.77e-07
coseno intra-brazo sobre 800 filas: Word2Vec +0.1598 (sd 0.2397) | aleatoria -0.0002 (sd 0.1003)
```

La segunda linea es la que da significado al efecto: el embedding tiene estructura de similitud
direccional real y la matriz aleatoria es isotropica. Lo que P1 mide es exactamente "aporta
informacion de similitud", que es lo que Word2Vec pretende hacer. No es un control de escala.

## 8. Contra el campeon

```text
A 0.583876    B 0.571365    C 0.659761    D 0.628797    V03 0.7772114525
gap del mejor brazo: 0.117450
```

**V03 remains the best verified leaderboard result.** Ese 0.117450 es un gap en la metrica local,
`f1_media_columnas`, que es la media del F1 positivo de HS, TR y AG. **No** es una distancia al
leaderboard: el scorer del host no esta verificado y HatEval Task B oficial reporta ademas Exact
Match Ratio, luego la equivalencia exacta no esta confirmada.

P78 no reentrena V03, no toca umbrales, no genera submission y no consulta el leaderboard.

## 9. Sesgo de diseno que hay que declarar

El pre-registro se escribio **sabiendo** que P73 y P74 estaban contaminados, y su rama favorable,
"el embedding aporta senal", era la esperable. Es sesgo de confirmacion a nivel de eleccion del
brazo primario, no de resultado. El antidoto esta en los datos: **la conclusion no depende del
brazo elegido**, porque P2, que es la condicion no alineada, la de P73, tambien es positiva
(+0.023307, 4 de 5, IC95 (0.006199, 0.039429)).

Por eso el primario se declaro como el unico confirmatorio y P2, P3 y P4 como diagnosticos
descriptivos, sin lectura inferencial y sin jerarquizacion.

## 10. Verificacion

- Las diez aserciones del pre-registro pasan. La 6 se cumple sobre **180 llamadas reales** a
  `partir_interna` capturadas parcheando `p75_01.partir_interna` alrededor de cada llamada a
  `epoca_estrella`, no sobre splits recalculados.
- El JSON se reproduce entero desde `reports/p78_pares.csv` y el OOF canonico de V03, con un
  calculo independiente: medias por brazo, deltas por semilla, deltas por comparacion, F1 por
  etiqueta e identidad de interaccion. Sin discrepancias.
- ASCII verificado byte a byte en dos pasadas: antes de construir el JSON, y sobre los
  temporales ya escritos.
- `reports/p78_manifiesto.sha256` verifica con `sha256sum -c` las diez piezas.
- Las seis unidades comunes entre la primera corrida y la final son **bit a bit identicas**, lo
  que demuestra que las tres rondas de correccion fueron de instrumentacion y de informe, y no de
  calculo.

## 11. Redaccion recomendada, sin sobre-afirmar

> Con early stopping estrictamente anidado y deltas pareados por semilla, el embedding Word2Vec
> mostro un efecto positivo frente a su control aleatorio de identica forma y norma
> (+0.024893 de macro-F1, positivo en los cinco pliegues, IC95 de bootstrap por clusters
> (0.007902, 0.040645)), aunque con una desviacion entre semillas de 0.021588 y un delta por
> semilla que va de +0.0428 a +0.0009, de modo que el intervalo que decide no cubre la variabilidad
> de reentrenamiento. El efecto de mayor tamano fue el de alinear la normalizacion del texto de
> entrada con la del corpus: +0.056969 en el brazo aleatorio, con desviacion entre semillas de
> 0.001272, y 65 por ciento de la ganancia total. V03 remains the best verified leaderboard result.
