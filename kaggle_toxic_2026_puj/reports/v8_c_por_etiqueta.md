# P5: constante de regularizacion distinta por etiqueta (V8-06)

Generado por `experiments/v8_06_c_por_etiqueta.py` (V8-06 (P5)).

## 1. El precedente y por que esta linea no es R1-C1

R1-C1 ajusto la constante y gano +0.0157 en local para perder 0.0108 en el externo.
Hizo dos cosas a la vez: una sola constante para las tres etiquetas, y elegirla contra
el mismo conjunto que despues evaluaba. **Aqui las dos se hacen bien**: la constante
se elige por etiqueta, y se elige con sub-pliegues de grupo propios dentro del tren de
cada pliegue, de modo que el conjunto de eleccion y el de medicion quedan separados.

Por eso el script mide ademas un **brazo de control** que usa una sola constante con el
mismo procedimiento interno. Sin ese brazo, cualquier diferencia observada no podria
atribuirse a la idea y no al protocolo.

## 2. Constantes elegidas en cada pliegue

| pliegue | constante global | HS | TR | AG |
|---|---|---|---|---|
| 1 | 2.0 | 4.0 | 2.0 | 4.0 |
| 2 | 4.0 | 2.0 | 4.0 | 1.0 |
| 3 | 2.0 | 1.0 | 2.0 | 4.0 |
| 4 | 8.0 | 1.0 | 8.0 | 1.0 |
| 5 | 8.0 | 4.0 | 8.0 | 4.0 |

Constantes elegidas sobre todo el desarrollo, que son las de la submission: HS 4.0, TR 2.0, AG 8.0

## 3. Resultados

| rama | metrica | diferencia | IC95 | pliegues | celdas | tasa | filtro | gate |
|---|---|---|---|---|---|---|---|---|
| anidada_por_etiqueta | 0.777518 | +0.000306 | [-0.00345, +0.00408] | 2/5 | 54 | 0.5185 frente a 0.6778 | no pasa | no supera |
| control_global_anidado | 0.775168 | -0.002043 | [-0.00660, +0.00214] | 1/5 | 53 | 0.4906 frente a 0.6811 | no pasa | no supera |

| rama | HS | TR | AG |
|---|---|---|---|
| V03 (referencia) | 0.7617 | 0.8157 | 0.7542 |
| anidada_por_etiqueta | 0.7602 | 0.8186 | 0.7538 |
| control_global_anidado | 0.7591 | 0.8186 | 0.7478 |

**La diferencia entre las dos ramas es el valor de la idea y no del protocolo**: elegir constante por etiqueta da +0.000306 y usar una sola da -0.002043.

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

- celdas que cambian: 123 de 3462
- filas que cambian: 106
- por etiqueta: HS 59, TR 20, AG 44

## 6. Decision y nota de presupuesto

Por D-063 el filtro de detectabilidad **deja de ser un bloqueante**: pasa a ser una
prediccion que se contrasta. La submission se escribe siempre que la linea vaya en
direccion positiva, y el envio es **siempre manual**.

La prediccion del filtro acertara: la tasa de acierto entre las celdas modificadas no llega al minimo, de modo que el externo deberia empeorar o no moverse.

Fecha: 2026-09-26T03:40:56+00:00. Duracion: 104.8 s.
