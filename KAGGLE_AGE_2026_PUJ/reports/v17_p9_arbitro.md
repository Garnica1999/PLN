# V17 · P9 — Experto selectivo con arbitro aprendido

## 1. Por que este experimento y no una compuerta mas

El diagnostico previo encontro la razon de fondo: **V12 esta mas confiado cuando se equivoca en la frontera** (separacion 0,809 en las filas que el especialista arregla frente a 0,539 en las que rompe), y las victorias del especialista caen fuera de la region de frontera (solo el 12,7 % y el 8,0 %). Por eso toda compuerta por ambiguedad falla: no hay banda de confianza que discrimine.

La pregunta bien planteada es otra: **dentro de la region de frontera, cuales de las ~11.000 filas convieneework al especialista**. Ahi la base honesta es -0,0080 (escuchar siempre) y el techo es +0,0107 (activar solo las que el especialista resuelve).

## 2. Diseno

- Un arbitro binario por frontera: decide si el especialista acierta esa fila.
- 47 entradas, **todas columnas derivadas que ya existian**: los votos y margenes de los 13 miembros, los 5 margenes de pares, los 3 de grupo, las 6 probabilidades de V12, las 6 de A2 y el acuerdo entre miembros. **Ninguna representacion nueva.**
- Protocolo anidado en tres niveles: el arbitro se ajusta en un bloque interior, el umbral se elige en **otro** bloque interior que el arbitro no ha visto, y despues se reentrena en todo el train exterior para aplicarse a la validacion exterior.
- Rejilla de umbrales: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] (pocos a proposito). Regularizacion C = 0.2.

## 3. Resultado

| Fold | V12 | Con arbitro | Diferencia |
|---|---|---|---|
| 1 | 0.6007 | 0.6005 | +0.0001 |
| 2 | 0.6050 | 0.6050 | +0.0000 |
| 3 | 0.6032 | 0.6030 | +0.0002 |
| 4 | 0.6004 | 0.6002 | +0.0002 |
| 5 | 0.5928 | 0.5928 | +0.0000 |

**Global: 0.6004 frente a 0.6005 de V12, delta -0.0001** · folds positivos 0/5 · IC95 pareado [-0.0002, -0.0001]

## 4. Activacion por fold y por frontera

| Fold | Frontera | Umbral | Activadas | Rescate | Dano | Neto |
|---|---|---|---|---|---|---|
| 0 | 18-24|25-34 | 0.70 | 780 | 1 | 5 | -4 |
| 0 | 25-34|35-49 | 0.90 | 1134 | 0 | 1 | -1 |
| 1 | 18-24|25-34 | 0.90 | 334 | 0 | 0 | +0 |
| 1 | 25-34|35-49 | 0.90 | 1099 | 0 | 0 | +0 |
| 2 | 18-24|25-34 | 0.60 | 818 | 1 | 9 | -8 |
| 2 | 25-34|35-49 | 0.90 | 1256 | 0 | 0 | +0 |
| 3 | 18-24|25-34 | 0.70 | 609 | 1 | 7 | -6 |
| 3 | 25-34|35-49 | 0.90 | 1221 | 0 | 0 | +0 |
| 4 | 18-24|25-34 | 0.90 | 327 | 0 | 0 | +0 |
| 4 | 25-34|35-49 | 0.90 | 1056 | 0 | 0 | +0 |

## 5. Gate de adopcion de V17

| Criterio | Valor | Cumple |
|---|---|---|
| efecto minimo | delta -0.0001 >= 0.003 | **no** |
| folds positivos | 0/5 >= 4 | **no** |
| significacion | IC95 [-0.0002, -0.0001] | **no** |

**Decision: NO ADOPTAR**

