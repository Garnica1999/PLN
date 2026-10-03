# P6: peso de clase distinto por etiqueta (V8-07)

Generado por `experiments/v8_07_peso_por_etiqueta.py` (V8-07 (P6)).

## 1. Que se busca y por que es el paso siguiente a P5

P5 mostro que elegir la constante por etiqueta vale +0,0023 sobre elegir una sola, y que
la tasa de acierto entre las celdas modificadas subio a 0,5185, **la primera vez por
encima del azar en la fase**. Eso rompe el patron de P1, P3 y P4, en el que las tres
lineas movian la decision hacia peor. Si el optimo de regularizacion difiere por
etiqueta, es razonable que el balanceo optimo tambien.

Con el balanceo de V03 el peso de cada clase es el cociente de filas entre el doble de
los positivos **de esa etiqueta**, de modo que ya hay un balanceo por etiqueta, pero
con el mismo criterio para las tres. Aqui se busca un **multiplicador por etiqueta**
sobre ese peso, con un intervalo alrededor de uno para poder apartarse del balanceo si
el dato lo pide. El multiplicador se aplica **sobre** el peso equilibrado y no en su
lugar, de modo que el valor uno reproduce exactamente el balanceo de V03.

## 2. Multiplicadores elegidos en cada pliegue

| pliegue | multiplicador global | HS | TR | AG |
|---|---|---|---|---|
| 1 | 1.0 | 1.5 | 1.0 | 1.0 |
| 2 | 3.0 | 1.0 | 3.0 | 1.0 |
| 3 | 1.0 | 0.75 | 1.0 | 0.75 |
| 4 | 2.0 | 2.0 | 2.0 | 1.0 |
| 5 | 3.0 | 1.0 | 3.0 | 1.0 |

Multiplicadores sobre todo el desarrollo, que son los de la submission: HS 1.0, TR 2.0, AG 1.0

## 3. Resultados

| rama | metrica | diferencia | IC95 | pliegues | celdas | tasa | filtro | gate |
|---|---|---|---|---|---|---|---|---|
| anidado_por_etiqueta | 0.771913 | -0.005298 | [-0.01210, +0.00191] | 1/5 | 148 | 0.3919 frente a 0.5649 | no pasa | no supera |
| control_global | 0.770354 | -0.006857 | [-0.01804, +0.00567] | 0/5 | 325 | 0.3508 frente a 0.5295 | no pasa | no supera |
| sin_equilibrar | 0.000000 | -0.777211 | [-0.79920, -0.75376] | 0/5 | 1683 | 0.2317 frente a 0.5057 | no pasa | no supera |

| rama | HS | TR | AG |
|---|---|---|---|
| V03 (referencia) | 0.7617 | 0.8157 | 0.7542 |
| anidado_por_etiqueta | 0.7572 | 0.8030 | 0.7556 |
| control_global | 0.7623 | 0.8030 | 0.7458 |
| sin_equilibrar | 0.0000 | 0.0000 | 0.0000 |

**El valor de la idea, separandola del protocolo**, es +0.001559: por etiqueta frente a un
multiplicador unico. Y **cuanto aporta el balanceo en si** lo dice la rama sin
equilibrar, que queda en -0.777211.

## 4. Integridad del fichero de envio

Resultado global: **todas pasan**.

| comprobacion | resultado | detalle |
|---|---|---|
| numero de filas | correcto | 1154 frente a 1154 en la prueba |
| nombres y orden de columnas | correcto | ['id', 'HS', 'TR', 'AG'] |
| valores de etiqueta | correcto | [0, 1] |
| sin valores ausentes | correcto | 0 celdas ausentes |
| sin identificadores duplicados | correcto | 0 repetidos |
| identificadores en el orden de la prueba | correcto | coinciden con el orden original |
| filas distintas del entrenamiento | correcto | 1731 frente a 1154 |

## 5. Cuanto se aparta de V03, celda a celda

- celdas que cambian: 139 de 3462, en 124 filas
- por etiqueta: HS 53, TR 50, AG 36

## 6. Decision y nota de presupuesto

Por D-063 el filtro de detectabilidad deja de ser un bloqueante y la submission se
escribe siempre que la linea vaya en direccion positiva. **El envio es manual.**

Fecha: 2026-09-26T03:43:15+00:00. Duracion: 88.6 s.
