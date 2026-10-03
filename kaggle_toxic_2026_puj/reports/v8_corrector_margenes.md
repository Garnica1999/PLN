# P7: corrector sobre los margenes y el desacuerdo entre pliegues (V8-08)

Generado por `experiments/v8_08_corrector_margenes.py` (V8-08 (P7)).

## 1. Por que no es el apilamiento que ya fallo

El apilamiento de la sonda v08 uso 44 columnas y dio 0.78163 en el externo frente a los
0.78306 de V03. Aqui la pregunta es mas estrecha: **los margenes de V03, y solo ellos,
¿llevan informacion que la decision de V03 no haya usado?** Un segundo modelo con
cuatro variables de entrada no es apilamiento, es un corrector, y la dimension manda en
lo que puede aportar.

## 2. El desacuerdo entre los cinco modelos, que es tambien la senal de P9

Si los cinco modelos coinciden mucho en una fila, la decision es segura; si discrepan,
es una fila limite. Ese desacuerdo es senal que V03 no usa y que se puede anadir como
columna sin tocar la representacion.

| etiqueta | desacuerdo medio |
|---|---|
| HS | 0.9175 |
| TR | 0.9671 |
| AG | 0.9256 |

Filas con las tres etiquetas por debajo de 0,4 de desacuerdo: **2 de
1483**, un 0.1 %.

## 3. Resultados del corrector

| rama | metrica | diferencia | IC95 | pliegues | celdas | tasa | filtro |
|---|---|---|---|---|---|---|---|
| corrector_margen | 0.775148 | -0.002063 | [-0.01089, +0.00705] | 0/5 | 252 | 0.3611 frente a 0.5381 | no pasa |
| corrector_margen_y_desacuerdo | 0.775148 | -0.002063 | [-0.01089, +0.00705] | 0/5 | 252 | 0.3611 frente a 0.5381 | no pasa |
| solo_desacuerdo | 0.417940 | -0.359271 | [-0.38528, -0.33291] | 0/5 | 2185 | 0.1648 frente a 0.5044 | no pasa |
| promedio_de_repeticiones | 0.782277 | +0.005066 | [-0.00371, +0.01414] | 4/5 | 182 | 0.5495 frente a 0.5527 | no pasa |

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

## 5. Cuanto se aparta de V03

- celdas que cambian: 192 de 3462, en 161 filas
- por etiqueta: HS 114, TR 19, AG 59

## 6. Decision

Por D-063 el envio es manual y el filtro es priorizacion, no bloqueante.

Fecha: 2026-09-26T03:49:43+00:00. Duracion: 110.6 s.
