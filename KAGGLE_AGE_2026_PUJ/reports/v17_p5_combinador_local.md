# V17 · P5 — Meta-combinador local por frontera

## 1. Que cambia respecto a P9

P9 preguntaba **a quien escucho** (seleccion entre dos modelos) y su respuesta fue que la frontera que hay que aprender no existe. P5 pregunta **que clase es** (combinacion), que es una hipotesis distinta: no hay dos modelos que elegir, hay una sola respuesta y la evidencia se mezcla.

## 2. Como se inyecta la evidencia

El combinador local da una probabilidad `q` de la clase superior del par para **toda** fila. Esa probabilidad **no se usa como decision**: se usa para **redistribuir la masa que V12 ya pone en las dos clases del par**, conservando el total. Por tanto (1) no puede mejorar la situacion de las clases faciles, porque no les mueve masa; (2) no puede empeorar las otras cuatro clases del par por entrar o salir de ellas; y (3) actua en todas las filas, no en una region activada.

La fuerza `w` va de 0 (V12 exacto) a 1 (sustitucion completa del reparto interno) y se elige en un bloque interior agrupado que el combinador no ha visto, por macro-F1 completa.

## 3. Resultado por fold

| Fold | V12 | Con combinador local | Diferencia |
|---|---|---|---|
| 1 | 0.6007 | 0.6007 | +0.0000 |
| 2 | 0.6050 | 0.6054 | -0.0003 |
| 3 | 0.6032 | 0.6032 | +0.0000 |
| 4 | 0.6004 | 0.5994 | +0.0010 |
| 5 | 0.5928 | 0.5924 | +0.0004 |

**Global: 0.6003 frente a 0.6005, delta -0.0002** · folds positivos 1/5 · IC95 pareado [-0.0005, +0.0001]

## 4. Doble metrica: las fronteras

| Frontera | V12 | Con combinador local |
|---|---|---|
| 18-24|25-34 | 0.4963 | 0.4953 |
| 25-34|35-49 | 0.4974 | 0.4968 |

## 5. Fuerza elegida por fold y frontera

| Fold | Frontera | Filas del par en train | Fuerza elegida | macro-F1 interior | sin inyeccion |
|---|---|---|---|---|---|
| 0 | 18-24|25-34 | 45,618 | 1.00 | 0.1996 | 0.199 |
| 0 | 25-34|35-49 | 45,261 | 0.25 | 0.206 | 0.206 |
| 1 | 18-24|25-34 | 45,618 | 1.00 | 0.1978 | 0.1973 |
| 1 | 25-34|35-49 | 45,262 | 0.25 | 0.2026 | 0.2025 |
| 2 | 18-24|25-34 | 45,619 | 0.75 | 0.1979 | 0.1966 |
| 2 | 25-34|35-49 | 45,262 | 0.00 | 0.2032 | 0.2032 |
| 3 | 18-24|25-34 | 45,619 | 1.00 | 0.1994 | 0.1989 |
| 3 | 25-34|35-49 | 45,262 | 0.75 | 0.2047 | 0.204 |
| 4 | 18-24|25-34 | 45,618 | 0.50 | 0.2025 | 0.202 |
| 4 | 25-34|35-49 | 45,261 | 0.50 | 0.2066 | 0.2058 |

## 6. Gate de adopcion de V17

| Criterio | Valor | Cumple |
|---|---|---|
| efecto minimo | delta -0.0002 >= 0.003 | **no** |
| folds positivos | 1/5 >= 4 | **no** |
| significacion | IC95 [-0.0005, +0.0001] | **no** |
| fronteras | 0/2 mejoran | **no** |

**Decision: NO ADOPTAR**

