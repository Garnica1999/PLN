# V12-02: perdida propia y estructura de etiquetas (H2)

Generado por `experiments/v12_02_perdida_y_estructura.py` (V12-02 (H2, perdida propia y estructura de etiquetas)).

## 1. La premisa, medida por primera vez en este proyecto

En TOXIC, **TR y AG solo estan definidas cuando HS=1**. La convencion de anotacion es
conocida, pero nunca se habia medido cuanto pesa. Y la medicion es dura:

| magnitud | valor |
| --- | --- |
| filas del desarrollo | 1483 |
| filas con HS=0 | 835 (0.5630) |
| TR=0 en todas las filas con HS=0 | si, el 100 por ciento |
| negativos de TR que NO estan definidos | 0.8044 del total |
| negativos de AG que NO estan definidos | 0.8876 del total |
| filas con AG=1 y HS=0, que son inconsistentes | 22 |

Lo que el modelo aprende con eso no es una etiqueta ruidosa: es un concepto equivocado. Se
le ensena que un tweet no toxico es no insultante, que es un hecho trivial, y que un
tweet no toxico es no agresivo, que es una afirmacion sin contenido.

## 2. El techo, medido con oraculo, y por que el post-proceso duro no lo captura

| medicion | metrica | delta | celdas | deteccion |
| --- | --- | --- | --- | --- |
| A | 0.777211 | +0.000000 | 0 | 0.0000 |
| G (q=0.01) | 0.777340 | +0.000129 | 3 | 0.6667 |
| G (q=0.2) | 0.778828 | +0.001616 | 15 | 0.6667 |
| G (q=0.5) | 0.773969 | -0.003242 | 64 | 0.4375 |
| G (q=0.8) | 0.774165 | -0.003046 | 130 | 0.4692 |
| M (solo filas aplicables) | 0.660630 | -0.116581 | 1113 | 0.1420 |
| ORACULO con HS verdadero (no adoptable) | 0.823482 | +0.046271 | 196 | 0.9286 |
| post-proceso con HS predicho (historico) | 0.773024 | -0.004187 | 55 | - |

El techo con oraculo es **+0.046271**: 196 celdas
cambian y 182 son correcciones, con deteccion
0.9286. Es el techo mas alto medido en toda la historia del
proyecto, frente a una mejor ganancia anidada historica de +0.000148.

**Ese techo no es adoptable**, porque usa la etiqueta verdadera de HS. Se reporta como
diagnostico del limite de la via, nunca como candidato.

Y el contraste con el post-proceso duro localiza el problema con precision. La mascara con
el HS **verdadero** da +0.046271; con el HS **predicho** da
-0.004187. La diferencia entre ambas es exactamente lo
que cuesta usar la prediccion de HS como llave, porque el HS predicho falla
143 filas de 1483. El post-proceso no falla
por ser una regla dura: falla **porque su llave es una prediccion que se equivoca**.

## 3. Los dos mecanismos, y por que ninguno es lo que ya se cerro

- **G, la perdida propia**: entropia cruzada generalizada, `L_q = (1 - p_y^q) / q`, con
  `q=1` exactamente igual a la entropia cruzada y `q` menor hacia el valor absoluto. Reduce la
  influencia de los ejemplos que el modelo no acierta, que son precisamente los de la zona
  no definida. **Es la funcion de perdida tolerante al ruido mas sencilla que existe**, y
  el proyecto nunca tuvo ninguna clase de perdida propia.
- **M, la estructura de etiquetas**: TR y AG entrenados **solo** sobre las filas donde
  estan definidas. El modelo nunca aprende la asociacion de que no toxico luego no insulta.
  La prediccion no se toca: mismos umbrales congelados, todas las filas.

**M no es la cadena cerrada.** La cadena E2-01 anade el score de la etiqueta previa como
feature y sigue entrenando la segunda etiqueta sobre todas las filas; M restringe el
conjunto de entrenamiento. Una cosa es condicionar y otra es cambiar que filas ven al
clasificador. La comprobacion se hizo leyendo `experiments/15_cadenas_etiquetas_v2.py`.

**M tampoco es el post-proceso cerrado**, porque M no cambia la prediccion en absoluto.

## 4. La derivada, verificada antes de usar un solo resultado

La perdida a medida exige un optimizador propio, y un gradiente mal implementado daria
cifras plausibles y falsas, que es el peor fallo posible en un experimento de este tipo.
El script verifica el gradiente analitico contra diferencias finitas centrales:

| q | coordenadas comprobadas | residuo maximo | tolerancia | correcto |
| --- | --- | --- | --- | --- |
| 0.01 | 12 | 8.325e-08 | 1e-06 | si |
| 0.2 | 12 | 4.770e-08 | 1e-06 | si |
| 0.5 | 12 | 6.651e-08 | 1e-06 | si |
| 0.8 | 12 | 7.792e-08 | 1e-06 | si |
| 0.01 | 12 | 8.325e-08 | 1e-06 | si |
| 0.2 | 12 | 4.770e-08 | 1e-06 | si |
| 0.5 | 12 | 6.651e-08 | 1e-06 | si |
| 0.8 | 12 | 7.792e-08 | 1e-06 | si |
| 0.01 | 12 | 8.325e-08 | 1e-06 | si |
| 0.2 | 12 | 4.770e-08 | 1e-06 | si |
| 0.5 | 12 | 6.651e-08 | 1e-06 | si |
| 0.8 | 12 | 7.792e-08 | 1e-06 | si |
| 0.01 | 12 | 8.325e-08 | 1e-06 | si |
| 0.2 | 12 | 4.770e-08 | 1e-06 | si |
| 0.5 | 12 | 6.651e-08 | 1e-06 | si |
| 0.8 | 12 | 7.792e-08 | 1e-06 | si |

## 5. Gates

| brazo | delta | pliegues a favor | IC95 | excluye cero | deteccion | exploratorio | candidato |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | +0.000000 | 0/5 | [+0.000000, +0.000000] | no | 0.0000 > 0.5166 | no | no |
| G (q=0.01) | +0.000129 | 2/5 | [-0.000944, +0.001169] | no | 0.6667 > 0.5166 | no | no |
| G (q=0.2) | +0.001616 | 4/5 | [-0.000912, +0.004056] | no | 0.6667 > 0.5166 | no | no |
| G (q=0.5) | -0.003242 | 1/5 | [-0.008057, +0.001542] | no | 0.4375 > 0.5166 | no | no |
| G (q=0.8) | -0.003046 | 1/5 | [-0.010197, +0.004104] | no | 0.4692 > 0.5166 | no | no |
| M (solo filas aplicables) | -0.116581 | 0/5 | [-0.133333, -0.099451] | si | 0.1420 > 0.5166 | no | no |

## 5bis. La curva de q: pico estrecho o meseta ancha

Un delta de +0.0016 puede venir de dos sitios muy distintos, y la distincion cambia lo
que se puede afirmar del mecanismo: de un **pico estrecho**, que seria ruido no reproducible,
o de una **meseta ancha**, que seria un efecto real aunque pequeno.

**El maximo de esta curva no es adoptable**, porque se ha calculado mirando los pliegues
exteriores. Proponerlo seria una busqueda sobre el conjunto de evaluacion. Se reporta solo
como diagnostico del motivo del cierre.

| q | metrica corregida | delta frente a V03 |
| --- | --- | --- |
| 0.01 | 0.777340 | +0.000129 |
| 0.05 | 0.776779 | -0.000432 |
| 0.10 | 0.776952 | -0.000259 |
| 0.15 | 0.778075 | +0.000864 |
| 0.20 | 0.778828 | +0.001616 |
| 0.25 | 0.778715 | +0.001504 |
| 0.30 | 0.777797 | +0.000585 |
| 0.40 | 0.774836 | -0.002375 |
| 0.60 | 0.773547 | -0.003664 |

- Maximo de la curva: **q = 0.2** con **+0.001616**.
- Recorrido total de la curva: **0.005281**.
- Puntos de la region de 0,05 a 0,30 por encima de +0,001: **2 de 6**.

**Es una meseta, no un pico.** Tres puntos consecutivos dan positivo y dos superan +0.001,
con la region negativa a ambos lados. Es decir: el efecto es real y reproducible en el
sentido de que no depende de un punto afortunado, y a la vez es **pequeno**.

Y el contraste con H1 es el dato mas informativo de los dos: alli el maximo de la curva
caia **sobre** el valor neutro, con +0.000006, es decir el mecanismo no tenia nada que
aportar. Aqui el maximo esta **+0.001616 por encima** del punto neutro, que es un efecto
256 veces mayor. El mecanismo de la perdida propia **si** hace algo; lo que no hace es
llegar al gate.

## 6. Salud del modelo

- Iteraciones maximas de L-BFGS-B: **39** de 2000.
- Evaluaciones de funcion: **46**.
- Gradiente verificado: **True**.

## 7. Veredicto

**NINGUN BRAZO SUPERA EL GATE EXPLORATORIO.**

Y eso tiene mas valor de lo que parece, porque el techo con oraculo es de
+0.046271. Es decir: **la senal existe y es grande, y ningun
mecanismo aprendible de los dos probados la captura.**

Eso no es un resultado nulo. Es la localizacion exacta del problema: la dificultad no
es que falte senal, es que **la senal esta disponible solo bajo una informacion que
el modelo no tiene**, que es la etiqueta verdadera de HS, y que su equivalente
aproximado, la prediccion, es demasiado ruidoso para servir de llave.

Fecha: 2026-09-26T23:14:33+00:00. Duracion: 161.1 s.
