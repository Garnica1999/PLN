# Track H: bagging con remuestreo por grupos (V6-06)

Generado por `experiments/v6_06_bagging_grupos.py` (V6-06 (H0-H1)).

## 1. Que se prueba y por que no es una repeticion de ENS

Los ensembles previos (ENS-01 a ENS-04) mezclaban **representaciones distintas** y no
superaron al mejor modelo. Aqui se promedian **V03-equivalentes**: el mismo pipeline
reentrenado sobre submuestras de grupos distintos. Son preguntas distintas, y el
gap analysis (V6-00) confirmo que no existe ningun bagging en el proyecto.

## 2. El techo se mide, no se supone

El track A (V6-03) midio la estructura de grupos:

- **1415 grupos distintos**, maximo 7 filas.
- **1378 de 1483 filas (92.9 %) en grupos
  de tamano 1.**

Eso tiene una consecuencia directa: **con casi todas las filas en grupos unicos,
remuestrar por grupos y remuestrar por filas son practicamente la misma cosa**, y el
promedio de modelos casi identicos no puede mover la metrica. Por eso este script
**mide la diversidad de los modelos que promedia** en vez de limitarse a informar un
delta. La magnitud que explica el resultado es la correlacion entre las
predicciones de cada par de modelos.

## 3. Resultados

| variante | corrected_cv | delta | IC95 | HS | TR | AG | salvage | damage |
|---|---|---|---|---|---|---|---|---|
| H0-control | 0.777211 | +0.0000 | [+0.0000, +0.0000] | 0.7617 | 0.8157 | 0.7542 | 0 | 0 |
| H1-bagging5 | 0.766688 | -0.0105 | [-0.0183, -0.0030] | 0.7519 | 0.8005 | 0.7477 | 1 | 11 |

## 4. Diversidad de los modelos que se promedian

| magnitud | valor |
|---|---|
| correlacion media entre pares | 0.957344 |
| correlacion minima | 0.937885 |
| diferencia media absoluta de probabilidad | 0.060600 |
| pares comparados | 50 |

## 5. Gate de dos niveles

| variante | delta >= +0.003 | IC95 excluye 0 | pliegues positivos | Gate 1 |
|---|---|---|---|---|
| H0-control | no (+0.0000) | no | 0/5 | no supera |
| H1-bagging5 | no (-0.0105) | no | 0/5 | no supera |

## 6. El resultado no es por falta de diversidad, y eso obliga a leerlo bien

La correlacion media entre pares de modelos es 0.9573, con una minima de
0.9379, y la diferencia media absoluta de
probabilidad es 0.0606. Hay **diversidad
real**: los cinco modelos no son el mismo modelo. Por tanto el empeoramiento **no se
explica por falta de diversidad**, que era la hipotesis que el track planteaba medir.

La explicacion coherente con los numeros es un **efecto de escala**: promediar cinco
modelos contrae las probabilidades hacia el centro, y los umbrales congelados de
V03 (0.44, 0.51, 0.46) estan calibrados para **un** modelo
individual. Un umbral que separa bien las probabilidades de un modelo puede no
separarlas igual tras el promedio.

### La tension que esto abre, y por que no se resuelve

Para evaluates el bagging con justicia habria que **reoptimizar los umbrales para el
promedio**. Pero eso rompe la comparabilidad con V03 y reproduce exactamente el
sesgo ya documentado en D-045: reoptimizar un umbral sobre el mismo OOF que se
evalua da un numero inflado (ahi, B0 pasaba de 0.7772 a 0.7808 sin cambiar el modelo).

En otras palabras: **no hay forma de evaluar el bagging de forma comparable con V03
sin reoptimizar umbrales, y reoptimizarlos rompe la comparacion**. El track se
cierra por una razon de **protocolo**, no porque se haya refutado un mecanismo. Se
deja escrito con esa precision para que la linea pueda reabrirse si algun dia se
dispone de una particion de calibracion separada del OOF de evaluacion.

## 7. Por que el control esta dentro del promedio y no aparte

El primer modelo de cada pliegue usa el **tren completo, sin remuestreo**. Asi H0 se
obtiene con `n_modelos = 1` sin cambiar una sola linea del pipeline, y H1 incluye
siempre un modelo equivalente a V03. Es la misma logica que los controles B0, A0, I0
y C1-0, aqui adaptada al promedio.

Fecha: 2026-09-26T05:52:43+00:00. Duracion: 70.8 s.
