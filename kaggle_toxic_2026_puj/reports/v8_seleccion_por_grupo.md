# Seleccion por grupo con criterio de informacion, y por correlacion (V8-12)

Generado por `experiments/v8_12_seleccion_por_grupo.py` (V8-12 (seleccion por grupo y por correlacion)).

## 1. El criterio que manda ahora

V8-11 demostro que la cifra de metrica local no basta: la rama de rasgos de
complejidad dio **+0,003493**, que supera el gate de +0,003, y **solo una de las cinco
repeticiones de la particion mejoro**. Una mejora que aparece en una de cinco
particiones es ruido de particion, y es exactamente lo que le paso a la sonda del
ensemble, que gano +0,005066 local y perdio 0,01184 en el externo. Por eso el informe
de cada rama lleva el **numero de repeticiones que mejoran** junto a la cifra.

## 2. Seleccion por grupo con criterio de informacion

El espacio se parte en los grupos que el material distingue: unigramas y bigramas de
palabra, y trigramas, tetragramas y pentagramas de caracter. Dentro de cada grupo se
conservan los rasgos de mayor informacion mutua con la combinacion de etiquetas. **La
diferencia con una seleccion global es que el criterio se aplica por grupo**, y por eso
un grupo pequeno no puede ser aniquilado por uno grande.

## 3. Eliminacion por correlacion

Se recorren los rasgos por magnitud descendente y se descarta el que esta correlacionado
con alguno ya conservado por encima de un umbral. Es la deteccion de redundancia que
describe el material, y reduce la dimension sin tocar la senal.

## 4. Resultados

| rama | rasgos | metrica | diferencia | IC95 | repeticiones que mejoran | celdas | tasa |
|---|---|---|---|---|---|---|---|
| grupo_10 | 2086 | 0.768673 | -0.008538 | [-0.02201, +0.00543] | 0/5 | 334 | 0.4162 frente a 0.5287 |
| grupo_25 | 5217 | 0.782535 | +0.005324 | [-0.00557, +0.01700] | 2/5 | 263 | 0.5057 frente a 0.5365 |
| grupo_50 | 10435 | 0.779890 | +0.002679 | [-0.00702, +0.01255] | 2/5 | 219 | 0.5023 frente a 0.5438 |
| correlacion_90 | 251 | 0.594095 | -0.183116 | [-0.20930, -0.15764] | 0/5 | 1353 | 0.2010 frente a 0.5071 |
| correlacion_95 | 252 | 0.590829 | -0.186382 | [-0.21289, -0.15998] | 0/5 | 1321 | 0.2006 frente a 0.5073 |
| sin_seleccion | 20870 | 0.781521 | +0.004309 | [-0.00482, +0.01331] | 0/5 | 183 | 0.5464 frente a 0.5525 |

| rama | repeticion 1 | repeticion 2 | repeticion 3 | repeticion 4 | repeticion 5 |
|---|---|---|---|---|---|
| grupo_10 | 0.766995 | 0.769503 | 0.770197 | 0.764745 | 0.761141 |
| grupo_25 | 0.776037 | 0.778048 | 0.777481 | 0.773127 | 0.777143 |
| grupo_50 | 0.776303 | 0.777623 | 0.775098 | 0.769266 | 0.779303 |
| correlacion_90 | 0.590070 | 0.575829 | 0.570572 | 0.577969 | 0.589205 |
| correlacion_95 | 0.585007 | 0.574238 | 0.565020 | 0.580933 | 0.576515 |
| sin_seleccion | 0.774589 | 0.775298 | 0.767069 | 0.772996 | 0.776803 |

## 5. Integridad del fichero de envio

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

## 6. Decision

La rama con mejor diferencia es **grupo_25**, con +0.005324,
y mejora en **2 de 5
repeticiones**.

**La submission se escribe pero no se recomienda enviarla**, con el mismo criterio que
V8-11: quince sondas pagadas, trece de catorce indistinguibles, y una mejora que no se
mantiene entre particiones no puede esperarse que se vea en el externo. **El envio es
manual por D-063.**

Fecha: 2026-09-26T04:33:59+00:00. Duracion: 975.1 s.
