# v6 P0-bis: interseccion de errores y estructura de etiquetas (V6-01)

Generado por `experiments/v6_01_estructura_errores.py` (V6-01 (P0-bis, ANALYSIS)).
No entrena nada: son evaluaciones downstream sobre el protocolo v2. Su funcion es
decidir si las lineas caras de la fase (D, E) merecen gastarse.

## 1. Estructura de etiquetas (diagnostico, no evidencia de mejora)

La especificacion pide P(TR|HS) y P(AG|HS) **como diagnostico** para saber si la
restriccion del dominio de entrenamiento de un modelo condicional es plausible.

| combinacion | n |
|---|---|
| `000` | 813 |
| `001` | 22 |
| `100` | 79 |
| `101` | 124 |
| `110` | 24 |
| `111` | 421 |

- P(TR=1 | HS=1) = **0.6867**
- P(TR=1 | HS=0) = **0.0**
- P(AG=1 | HS=1) = **0.841**
- P(AG=1 | HS=0) = **0.0263**

**Hallazgo 1: la implicacion TR implica HS es determinista en los datos.** Hay
0 casos de TR=1 con HS=0 frente a
445 de TR=1 con HS=1. No es una
tendencia, es una invariante del corpus. Eso **valida la factorizacion del track B**
en el sentido de que es una restriccion real que un clasificador TR independiente no
puede expresar: si V03 predice TR=1 con HS=0 en alguna fila, la prediccion es
logicamente imposible segun los datos.

**Hallazgo 2: la rama AG|HS=0 es casi nula y muy pequena.** Ocurre en
22 ejemplos con P = 0.026, frente a 0.841 en la
rama HS=1. Un condicional de AG necesita las dos ramas, y la segunda no tiene datos
para aprenderla: eso justifica la forma regularizada B2r con shrinkage o
prior fuerte, no un modelo libre.

## 2. Interseccion de errores: ¿hay informacion que V03 no usa?

V03 acierta **340 de 1483 filas**. Para cada candidato se
cuentan cuatro categorias. La que decide es la **segunda**: filas donde V03 falla y
el candidato acierta. Ahi la informacion existe y solo hace falta incorporarla.

| candidato | V03 acierta, otro falla | V03 falla, otro acierta | ambos | ninguno | correccion neta | info. disponible |
|---|---|---|---|---|---|---|
| M3-SVC | 13 | 10 | 327 | 1133 | -3 | 0.87 % |
| M4-CHAR-LR | 20 | 12 | 320 | 1131 | -8 | 1.05 % |
| M1-WORD-LR | 6 | 9 | 334 | 1134 | +3 | 0.79 % |

El mejor es **M4-CHAR-LR** con 12 filas que
V03 falla y el candidato acierta. Ese es el techo de lo que un modelo local puede
rescatar **si** se combinara bien: es informacion que ya existe en el corpus y que el
modelo de referencia no esta usando.

## 3. Lectura para decidir el gasto (cuantitativa)

El mejor candidato recupera **12 filas de 1483**,
un **0.81 %**. Eso es una cota
**superior** de lo que cualquier linea que anada otro modelo podria ganar, y no un
resultado: combinar dos clasificadores elige entre sus aciertos, no los suma.

Cota superior generosa para dimensionarla: si las 12 filas cambiaran una sola etiqueta y todas se
ganaran sin romper ninguna otra, la metrica (media de tres columnas sobre 1483 filas) subiria como mucho
**+0.0027**, que esta **por debajo del
gate de +0.003** de la fase.

**Consecuencia directa y definitiva**: las lineas cuyo techo es "recuperar filas que
V03 falla" no pueden superar el gate, por pequena que sea la interseccion. Eso
cierra **D** (TextCNN) y **E** (kNN) como vias de mejora, y es una decision de
presupuesto, no un resultado negativo: no se ejecutan porque su techo teorico esta
por debajo del umbral de adopcion, no porque hayan fallado.

Las lineas que **si** tienen margen son las que cambian el propio V03 (B, A, I) o
reducen su varianza (H), porque no dependen de arar filas ajenas sino de acertar
mas filas de las que V03 ya acierta.

## 4. Advertencia

El numero de filas que un candidato acierta y V03 falla **no es la mejora esperada**:
combinar dos modelos no suma aciertos, elige. Es un techo de informacion, no un
resultado. El criterio de adopcion sigue siendo el de la puerta de dos niveles
(delta y IC95 y folds), y la interseccion solo decide si vale la pena abrir la linea.

Fecha: 2026-09-26T01:20:30+00:00. Duracion: 5.3 s.
