# P60-D - Numero efectivo de muestras

Generado por `experiments/p60_04_distribucion.py`. Fecha 2026-09-28T15:28:28+00:00.

## Motivacion e hipotesis

V03 corrige la frecuencia con `class_weight='balanced'`, proporcional a `1 / (2 * pi_c)`.
Cui et al. proponen otra FUNCION del mismo recuento, el **numero efectivo de muestras**,
que satura: una clase muy grande pesa proporcionalmente a 1 en vez de a `1 / n_c`.

## Por que no es V8-07

`V8-07` probo QUE peso por etiqueta, con -0,005298 y 1 de 5 pliegues. Eso es el eje de
la eleccion del peso. Esta pista prueba que FORMA de `n_c` lo produce, y el gap check de
la familia devuelve 0 filas, luego no es duplicacion por nombre.

## Normalizacion, que es lo que hace la comparacion controlada

Ambas formas se reescalan para que la **media ponderada del peso sea 1**, que es lo que ya
cumple la forma de V03, porque `n_pos * w_pos + n_neg * w_neg = n`. Sin eso los dos brazos
diferirian en la masa total de la perdida y por tanto en la regularizacion efectiva, que es
exactamente el error que invalido la primera version de la pista L.

## Configuracion, fijada antes de mirar validation

```text
beta   0.9999, constante del metodo, una sola forma, sin rejilla
forma  (1 - beta) / (1 - beta ** n_c), normalizada a media ponderada 1
resto  representacion de V03, C = 2,0, umbrales 0,44 / 0,51 / 0,46
```

## La prediccion cuantitativa, medida antes de entrenar

| pliegue | etiqueta | prevalencia | razon balanceada | razon numero efectivo | diferencia |
|---:|---|---:|---:|---:|---:|
| 1 | HS | 0.4376 | 1.285164 | 1.275782 | -0.730 por ciento |
| 1 | TR | 0.3010 | 2.322129 | 2.268495 | -2.310 por ciento |
| 1 | AG | 0.3820 | 1.618102 | 1.595826 | -1.377 por ciento |
| 2 | HS | 0.4368 | 1.289575 | 1.280034 | -0.740 por ciento |
| 2 | TR | 0.2993 | 2.340845 | 2.286326 | -2.329 por ciento |
| 2 | AG | 0.3820 | 1.618102 | 1.595826 | -1.377 por ciento |

Diferencia relativa maxima: **2.329 por ciento**, media -1.477.

## Control

```text
brazo 0 (forma balanceada de V03)  corrected_cv 0.7772647318762402
  delta vs V03 +0.0000532794 | celdas distintas 3
```

## Resultados

```text
brazo 0 corrected_cv 0.7772647319
brazo 1 corrected_cv 0.7768328831
delta PONDERADO PURO -0.0004318488
delta vs V03         -0.0003785694
pliegues que mejoran 0/5
rescate 3 | dano 3 | neto +0 | cambian 6
coseno medio entre las dos soluciones 0.9999996203 (min 0.99999909, max 0.99999994)
convergencia True
```

## Gate

| criterio | valor | superado |
|---|---|---|
| efecto del ponderado >= +0,003 | -0.000432 | **False** |
| al menos 4 de 5 pliegues | 0/5 | **False** |
| gate completo | | **False** |

Bootstrap NO ejecutado: el gate no se supero.

## Decision

```text
decision  CLOSED. La forma de numero efectivo es practicamente la misma que la balanceada de V03 en este corpus, con diferencia relativa de la razon por debajo del dos por ciento
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_04_distribucion.py`. Sin push.*