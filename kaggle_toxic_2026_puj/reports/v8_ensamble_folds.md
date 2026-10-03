# P1: promedio de los cinco modelos de pliegue al predecir (V8-01)

Generado por `experiments/v8_01_ensamble_folds.py` (V8-01 (P1)).

## 1. Que es y por que quedaba pendiente de la fase v7

Cinco modelos V03 equivalentes, cada uno entrenado sobre cuatro de los cinco
pliegues del protocolo oficial. Para el conjunto de prueba se usa **la media
aritmetica de las cinco probabilidades**, etiqueta a etiqueta. Misma representacion,
mismos hiperparametros, mismos umbrales congelados de 0.44, 0.51 y 0.46. Sin
metamodelo y sin ajustar pesos.

Esta linea quedo abierta en la fase v7 por una razon concreta: **su comportamiento
externo no lo captura el protocolo local**. El desarrollo ya tiene una prediccion de
cada fila hecha por el modelo que no la entreno, de modo que reconstruir ese mismo
promedio sobre el desarrollo no aporta informacion nueva. Lo unico que queda por medir
es lo que ocurre sobre la prueba, y eso exige enviar.

## 2. Control de calidad del codigo

El control se hace contra el **OOF canonico guardado en disco**, fila a fila, y no
solo contra la cifra de la metrica. Es una comprobacion de identidad de
entrenamiento: si las probabilidades coinciden celda a celda, el pipeline es
exactamente el de V03 y ninguna cifra posterior puede deberse al codigo.

- metrica del promedio por pliegue reconstruido: **0.777211**
- metrica del OOF canonico cargado: 0.777211
- referencia de V03: 0.777211, con tolerancia declarada de 0.000500
- diferencia maxima por celda: 0.000e+00
- identificadores en el mismo orden: True

## 3. Los cinco modelos

| modelo | filas de entrenamiento | filas de validacion | metrica en su pliegue |
|---|---|---|---|
| 1 | 1186 | 297 | 0.796010 |
| 2 | 1186 | 297 | 0.794034 |
| 3 | 1187 | 296 | 0.778860 |
| 4 | 1187 | 296 | 0.738598 |
| 5 | 1186 | 297 | 0.779056 |

Cada modelo se ajusto **una sola vez** y ese mismo objeto predijo su pliegue de
validacion y el conjunto de prueba. Es la unica forma de que las dos columnas sean
comparables: se ajustan 5 modelos sobre las filas del desarrollo, sin
repetir ningun ajuste.

## 4. Efecto de escala: el riesgo conocido de esta linea

El promedio **contrae las probabilidades hacia el centro**. Ese es el riesgo que
PROVIDE-03 describio en el track H de la fase v6: los umbrales congelados de V03
estan calibrados para **un** modelo individual, y un umbral que separa bien las
probabilidades de un modelo puede no separarlas igual despues del promedio. Por eso
el efecto se mide y no se supone.

Sobre la prueba, el promedio baja la media de las probabilidades en
+0.001334 respecto del modelo completo, y su desviacion tipica sobre el
desarrollo es 0.295165.

**La metrica local de esta linea no es informativa, y conviene decirlo con
claridad**: la columna de desarrollo es el promedio por pliegue, que es la definicion
del OOF canonico, asi que tiene que coincidir con V03 y asi coincide. Lo unico nuevo
es el promedio sobre la prueba, y eso solo se mide en el externo. Las cifras que
siguen se registran para dejar constancia, no para decidir nada.

Una cifra que **no** debe compararse con la referencia es la metrica de
resubstitucion del modelo completo, 0.969085: ese modelo se
evalua sobre las mismas filas con las que se ajusto, de modo que la cifra esta
inflada por construccion y no dice nada sobre su capacidad.

| etiqueta | F1 del promedio por pliegue | F1 del modelo completo en resubstitucion |
|---|---|---|
| HS | 0.7617 | 0.9685 |
| TR | 0.8157 | 0.9681 |
| AG | 0.7542 | 0.9707 |

## 5. Desplazamiento de la prediccion sobre la prueba

La comparacion honesta es contra el modelo completo **sobre la misma prueba**, porque
las dos columnas salen de la misma fila. Si el promedio contrae las probabilidades, la
tasa de positivos baja respecto del modelo completo, y eso es exactamente lo que
indica la ultima columna.

| etiqueta | promedio | modelo completo | entrenamiento | diferencia promedio menos completo |
|---|---|---|---|---|
| HS | 0.4376 | 0.4428 | 0.4370 | -0.0052 |
| TR | 0.2834 | 0.2842 | 0.3001 | -0.0009 |
| AG | 0.3553 | 0.3605 | 0.3823 | -0.0052 |

Nota sobre la lectura de esta tabla: la prueba tiene 1154 filas y su composicion
no es necesariamente la del desarrollo, de modo que la tasa de positivos de la prueba
no se compara con la del entrenamiento. La comparacion valida es la de las dos
primeras columnas.

## 6. Verificacion de integridad del fichero

Resultado global: **todas las comprobaciones pasan**.

| comprobacion | resultado | detalle |
|---|---|---|
| numero de filas | correcto | 1154 filas en el fichero frente a 1154 en la prueba |
| nombres y orden de columnas | correcto | columnas ['id', 'HS', 'TR', 'AG'] frente a las esperadas ['id', 'HS', 'TR', 'AG'] |
| valores de etiqueta | correcto | valores distintos: [0, 1] |
| sin valores ausentes | correcto | 0 celdas ausentes |
| sin identificadores duplicados | correcto | 0 identificadores repetidos |
| identificadores en el orden del conjunto de prueba | correcto | coinciden con el orden original |
| numero de filas distinto del entrenamiento | correcto | entrenamiento 1731 frente a prueba 1154 |
| etiquetas dentro de cero y uno | correcto | todas las celdas de etiqueta estan en el intervalo |
| sin cadenas vacias | correcto | 0 identificadores vacios |

## 7. Que queda pendiente de la decision

**El fichero esta escrito y verificado, y no se envia automaticamente.** El
resultado de la sonda decide la bifurcacion de la fase:

- Si el externo mejora de forma material, se congela la arquitectura y se ejecutan
  como maximo dos lineas mas, que son el desacuerdo entre pliegues y la segunda
  semilla.
- Si no mejora, esta linea se cierra **sin variantes** y se entra por la estabilidad
  de features, que es la forma barata de reducir la dimension sin perder senal.

No se harn variantes de umbral, de pesos ni de semilla antes de conocer ese numero:
la fase v6 demostro que ajustar la decision contra el mismo conjunto no transfiere.

Fecha: 2026-09-26T02:52:57+00:00. Duracion: 8.0 s.
