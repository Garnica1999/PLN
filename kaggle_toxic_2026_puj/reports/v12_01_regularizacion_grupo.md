# V12-01: regularizacion por etiqueta y por grupo (H1)


Generado por `experiments/v12_01_regularizacion_grupo.py` (V12-01 (H1, regularizacion por etiqueta y por grupo)).


## 1. La hipotesis, y por que no es un hiperparametro mas


> La misma caracteristica puede tener utilidad diferente para HS, TR y AG, y compartir
> exactamente la misma estructura de regularizacion puede limitar una o mas etiquetas.
> exactamente la misma estructura de regularizacion puede limitar una o mas etiquetas.

El precedente mas cercano es **V8-06**, que ajusto una constante distinta **por etiqueta**
El precedente mas cercano es **V8-06**, que ajusto una constante distinta **por etiqueta**
sobre el espacio entero. Aqui lo que se varya es **la constante del grupo de
caracteristicas dentro de la etiqueta**.
caracteristicas dentro de la etiqueta**.

Y la razon por la que es un mecanismo y no otro valor de un parametro es aritmetica. Para
Y la razon por la que es un mecanismo y no otro valor de un parametro es aritmetica. Para
una regresion logistica con penalizacion L2, sustituir `X` por `X * diag(s)` produce, en
terminos de los coeficientes originales, una penalizacion
terminos de los coeficientes originales, una penalizacion

    (1 / 2C) * sum_j s_j^2 * w_j^2
    (1 / 2C) * sum_j s_j^2 * w_j^2

Es decir, **multiplicar el bloque j por `s_j` multiplica su peso de penalizacion efectivo
Es decir, **multiplicar el bloque j por `s_j` multiplica su peso de penalizacion efectivo
por `s_j^2`**. No cambia la clase de modelo, no cambia la dimension, no anade variables, y
se hace con el `LogisticRegression` de la biblioteca sin tocar su optimizador.
se hace con el `LogisticRegression` de la biblioteca sin tocar su optimizador.

## 2. Porque hay un solo grado de libertad, y no dos
## 2. Porque hay un solo grado de libertad, y no dos

Escalar el bloque de palabra por `a` y el de caracter por `b` es lo mismo, salvo una
Escalar el bloque de palabra por `a` y el de caracter por `b` es lo mismo, salvo una
constante global, que fijar `a=1` y `b=s`. Y como la matriz se renormaliza para conservar
su norma de Frobenius, esa constante global queda fijada. El mecanismo tiene **un solo
su norma de Frobenius, esa constante global queda fijada. El mecanismo tiene **un solo
grado de libertad identificable: el cociente entre los dos bloques**.


Eso deja **tres cocientes y cuatro constantes**, con el cociente neutro en 1.0, que
reproduce el control. Tres puntos no son una rejilla, y el central hace funcionar la
reproduce el control. Tres puntos no son una rejilla, y el central hace funcionar la
comprobacion interna.


La renormalizacion resuelve ademas un problema de comparacion: sin ella, un resultado
negativo significaria dos cosas indistinguibles, que el mecanismo no ayuda o que el punto
negativo significaria dos cosas indistinguibles, que el mecanismo no ayuda o que el punto
de operacion se movio. La tasa de positivos predicha se registra como diagnostico.


## 3. Los cuatro brazos, y por que hacen falta cuatro


- **A**: V03 exacto, constante 2.0 y sin escalado. Control de identidad
- **B**: constante distinta por etiqueta, sin escalado. Es el mecanismo de V8-06 y sirve de control para no atribuir a la estructura de grupo lo que sea del ajuste por etiqueta
- **C**: constante por etiqueta y cociente entre bloques por etiqueta. La variante
- **D**: constante por etiqueta y un unico cociente compartido por las tres. Diagnostico: si C y D coinciden, la parte por etiqueta no aporta nada

Un solo brazo no permitiria atribuir nada. **B** responde si la ganancia viene del ajuste
Un solo brazo no permitiria atribuir nada. **B** responde si la ganancia viene del ajuste
por etiqueta, y **D** si la parte por etiqueta aporta algo: si C y D coinciden, el efecto
es un cociente de bloques mejor elegido y no una asignacion distinta por etiqueta.
es un cociente de bloques mejor elegido y no una asignacion distinta por etiqueta.

## 4. Identidad del control
## 4. Identidad del control

El brazo A da **0.7772114525165721**, con diferencia
El brazo A da **0.7772114525165721**, con diferencia
**0.000e+00** respecto a la
referencia. La identidad del pipeline se confirma antes de mirar ningun resultado.
referencia. La identidad del pipeline se confirma antes de mirar ningun resultado.

Identidad interna: C eligio el cociente neutro en **0 de 5** pliegues, y en esos pliegues tiene que igualar a B.
igualar a B.
igualar a B.

## 5. Resultados
## 5. Resultados

| brazo | metrica corregida | metrica corregida | delta vs A | HS F1 | HS F1 | TR F1 | AG F1 | AG F1 | celdas que cambian | deteccion | deteccion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.777211 | referencia | referencia | 0.7617 | 0.8157 | 0.8157 | 0.7542 | 0 | 0 | - |
| B | 0.777628 | +0.000417 | +0.000417 | 0.7626 | 0.8198 | 0.8198 | 0.7504 | 30 | 30 | 0.5000 |
| C | 0.779949 | +0.002738 | +0.002738 | 0.7619 | 0.8180 | 0.8180 | 0.7600 | 102 | 102 | 0.5196 |
| D | 0.777847 | +0.000636 | +0.000636 | 0.7641 | 0.8201 | 0.8201 | 0.7493 | 111 | 111 | 0.4775 |

### Delta por pliegue
### Delta por pliegue

| brazo | pliegue 1 | pliegue 2 | pliegue 3 | pliegue 4 | pliegue 5 | a favor |
| --- | --- | --- | --- | --- | --- | --- |
| B | -0.002348 | +0.002146 | +0.001468 | +0.002103 | -0.001927 | 3/5 |
| C | -0.000974 | +0.000212 | +0.003226 | +0.012705 | -0.001927 | 3/5 |
| D | -0.000974 | +0.001423 | +0.006730 | -0.002670 | -0.001927 | 2/5 |

### Seleccion que hizo la validacion anidada
### Seleccion que hizo la validacion anidada

| brazo | constantes por pliegue | cocientes por pliegue |
| --- | --- | --- |
| B | [[2.0, 2.0, 4.0], [1.0, 4.0, 2.0], [2.0, 4.0, 1.0], [4.0, 2.0, 2.0], [2.0, 2.0, 4.0]] | todos 1.0 |
| C | [[1.0, 4.0, 4.0], [1.0, 1.0, 4.0], [2.0, 4.0, 2.0], [2.0, 2.0, 4.0], [2.0, 2.0, 4.0]] | [[1.6667, 1.6667, 1.6667], [1.0, 0.6, 1.6667], [1.6667, 0.6, 1.6667], [0.6, 0.6, 1.6667], [1.6667, 1.0, 1.0]] |
| D | [[1.0, 4.0, 4.0], [1.0, 1.0, 4.0], [2.0, 4.0, 2.0], [2.0, 2.0, 4.0], [2.0, 2.0, 4.0]] | [[1.6667, 1.6667, 1.6667], [0.6, 0.6, 0.6], [1.6667, 1.6667, 1.6667], [0.6, 0.6, 0.6], [1.0, 1.0, 1.0]] |

### Tasa de positivos predicha, el diagnostico del punto de operacion
### Tasa de positivos predicha, el diagnostico del punto de operacion

| brazo | HS | TR | AG |
| --- | --- | --- | --- |
| A | 0.4572 | 0.3001 | 0.3776 |
| B | 0.4579 | 0.2987 | 0.3796 |
| C | 0.4693 | 0.3001 | 0.3790 |
| D | 0.4720 | 0.3034 | 0.3790 |

El umbral de detectabilidad de V8-02 es **0.5166**. Ganar por debajo de ese minimo
El umbral de detectabilidad de V8-02 es **0.5166**. Ganar por debajo de ese minimo
no es ganar: es cambiar la prediccion sin la senal que la respalde.


## 6. Gates


| brazo | delta | delta | pliegues a favor | IC95 | IC95 | IC95 excluye cero | deteccion | deteccion | exploratorio | candidato | candidato |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | +0.000417 | 3/5 | [-0.002604, +0.003520] | no | no | 0.5000 > 0.5166 | no | no | no |
| C | +0.002738 | 3/5 | [-0.003420, +0.008848] | no | no | 0.5196 > 0.5166 | no | no | no |
| D | +0.000636 | 2/5 | [-0.006456, +0.007424] | no | no | 0.4775 > 0.5166 | no | no | no |

## 6bis. Por que falla el gate: la curva del cociente
## 6bis. Por que falla el gate: la curva del cociente

El gate falla por **inconsistencia entre pliegues**, y eso admite dos explicaciones que
El gate falla por **inconsistencia entre pliegues**, y eso admite dos explicaciones que
conducen a cierres muy distintos: que el mecanismo no sirva, o que mi manera de elegir el
cociente fuera ruidosa. La curva del cociente las separa.
cociente fuera ruidosa. La curva del cociente las separa.

Se mide con la **constante fija de V03** y el cociente aplicado a las tres etiquetas, sin
Se mide con la **constante fija de V03** y el cociente aplicado a las tres etiquetas, sin
seleccion anidada. Su unico proposito es ver si la curva tiene un maximo separado del valor
neutro.
neutro.

| cociente | metrica corregida | delta frente al neutro |
| --- | --- | --- |
| 0.400 | 0.764842 | -0.012370 |
| 0.600 | 0.771666 | -0.005546 |
| 0.800 | 0.774484 | -0.002727 |
| 1.000 | 0.777211 | +0.000000 |
| 1.250 | 0.777217 | +0.000006 |
| 1.667 | 0.774676 | -0.002536 |
| 2.500 | 0.764932 | -0.012279 |

- Maximo de la curva: cociente **1.25**, con **+0.000006** sobre el neutro.
- Maximo de la curva: cociente **1.25**, con **+0.000006** sobre el neutro.
- Recorrido total de la curva: **0.012375**.
- Cocientes que baten al neutro: **1** de 6.
- Cocientes que baten al neutro: **1** de 6.

**El maximo cae sobre el valor neutro.** El cociente tiene un efecto real y grande sobre
**El maximo cae sobre el valor neutro.** El cociente tiene un efecto real y grande sobre
la metrica, con un recorrido de 0.012375, pero ese
efecto esta enteramente capturado por el punto de no hacer nada, y alejarse de el en
efecto esta enteramente capturado por el punto de no hacer nada, y alejarse de el en
cualquier direccion empeora. Es decir: **la estructura de penalizacion de V03 ya esta en su
optimo**, y el optimum del mecanismo es el punto neutro.
optimo**, y el optimum del mecanismo es el punto neutro.

Eso convierte el cierre en algo mucho mas fuerte que "el gate no se supero": la razon de
que no se supere es que **no hay nada que extraer**, y no que mi seleccion fuera ruidosa.


**Aviso obligatorio**: el maximo de esta curva **no es adoptable**, porque se ha calculado
mirando los pliegues exteriores. Proponerlo seria una busqueda sobre el conjunto de
mirando los pliegues exteriores. Proponerlo seria una busqueda sobre el conjunto de
evaluacion, que es exactamente lo que el gate exploratorio existe para impedir. Se reporta
solo como diagnostico del motivo del cierre.
solo como diagnostico del motivo del cierre.

## 7. Salud del modelo
## 7. Salud del modelo

- Iteraciones maximas: **6** de 2000.
- Iteraciones maximas: **6** de 2000.
- Advertencias: **0**. El ajuste se detiene si aparece alguna,
  porque una advertencia de convergencia invalidaria la comparacion entre brazos.
  porque una advertencia de convergencia invalidaria la comparacion entre brazos.
- Diagnostico: **ajuste sano: ninguna advertencia y ninguna convergencia al limite**.


## 8. Veredicto


**NINGUN BRAZO SUPERA EL GATE EXPLORATORIO, Y LA CURVA DIAGNOSTICA DICE POR QUE.**


El brazo C llega a +0.002738, que supera el minimo de
0.002, pero solo gana en 3/5 de
0.002, pero solo gana en 3/5 de
5 pliegues y su intervalo pareado incluye el cero. Sin ese pliegue favorable el delta
cae a +0.000134, es decir practicamente a nada.
cae a +0.000134, es decir practicamente a nada.

La curva del cociente con constante fija pone el **maximo en 1.25**, con
La curva del cociente con constante fija pone el **maximo en 1.25**, con
**+0.000006** sobre el valor neutro. Ese es el dato que cierra la linea:
**el optimo del mecanismo es el punto de no hacer nada**, luego la estructura de
**el optimo del mecanismo es el punto de no hacer nada**, luego la estructura de
penalizacion entre bloques de V03 ya esta en su optimo y no hay holgura que
extraer. No es que la seleccion fuera ruidosa: es que no hay nada que seleccionar.
extraer. No es que la seleccion fuera ruidosa: es que no hay nada que seleccionar.

Lo que deja el experimento es mas util que un resultado negativo generico. La
Lo que deja el experimento es mas util que un resultado negativo generico. La
comparacion entre B, C y D separa **cuanto de cualquier ajuste por etiqueta**
(+0.000417) y **cuanto de la estructura de grupo**
(+0.000417) y **cuanto de la estructura de grupo**
(+0.002738), y esa separacion no existia antes. Ademas D
(+0.000636) muestra que la parte por etiqueta del cociente no
(+0.000636) muestra que la parte por etiqueta del cociente no
aporta: el efecto, si lo hubiera, seria un cociente comun.


**Se cierra la linea H1.** Y con ella la parte de la hipotesis que decia que la
dificultad estaba en la estructura de regularizacion: para el reparto entre bloques de
dificultad estaba en la estructura de regularizacion: para el reparto entre bloques de
caracteristicas, la estructura no era el cuello de botella.

Fecha: 2026-09-26T22:53:50+00:00. Duracion: 645.3 s.
