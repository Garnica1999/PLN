# P66 - Complementariedad, rescate de alta confianza y bootstrap

Generado por `experiments/p66_03_complementariedad_bootstrap.py`. Fecha 2026-09-29T15:52:07+00:00.

## Las cuatro preguntas que quedaban abiertas

1. **P14** una segunda fuente anade algo, y solo se mide si ambas tienen rescate unico
2. **P17** cuantos errores de V03 rescata y cuantas filas correctas dana, por etiqueta
   y por estrato de longitud
3. **P18** los errores rescatados son los que V03 fallo **con confianza**, que es la
   pregunta mas afilada de la fase: una fusion que solo rescata filas pegadas al
   umbral ha encontrado calibracion, no representacion
4. **P21** bootstrap con la unidad de remuestreo del proyecto

## La unidad de remuestreo

1415 grupos, desde holdout_v2_ids.csv, columna grupo_near_dup_autor. Se remuestrea el **grupo** y no la fila, porque dentro de un grupo las filas no son independientes y remuestrear filas subestimaria el intervalo.

## La correccion residual de MI_arbol

```text
real        0.7798857693  delta +0.0026743167
barajado    0.7757724104  delta -0.0014390421
brecha      +0.0041133588
```

## El rescate unico de cada candidato

La regla es anterior a la ejecucion: dos fuentes se combinan **solo** si cada una tiene rescate unico. Una segunda fuente que corrige las mismas celdas no es una segunda fuente, es la misma con ruido.

| candidato | rescate unico sobre MI_arbol | autorizado a combinar |
|---|---:|---|
| negacion | 83 | True |
| word_1_3 | 67 | True |
| cohesion | 75 | True |
| huecos_1_2 | 98 | True |
| contraste | 53 | True |
| huecos_6_10 | 74 | True |

## Rescate, dano y alta confianza

| corte | rescate | dano | neto | rescate de alta confianza | cerca del umbral |
|---|---:|---:|---:|---:|---:|
| HS | 13 | 10 | +3 | 0 | 13 |
| TR | 16 | 16 | +0 | 4 | 12 |
| AG | 24 | 27 | -3 | 2 | 22 |

| estrato | rescate | dano | neto | F1 fusion | F1 V03 |
|---|---:|---:|---:|---:|---:|
| CORTA | 1 | 3 | -2 | 0.9452 | 0.9480 |
| MEDIA | 21 | 22 | -1 | 0.7944 | 0.7964 |
| LARGA | 26 | 17 | +9 | 0.7047 | 0.6834 |
| MUY_LARGA | 5 | 11 | -6 | 0.5653 | 0.5606 |

## El bootstrap

```text
remuestreos validos   2000
delta medio           +0.0027270194
IC95                  [-0.0044311634, +0.0102108745]
incluye el cero       True
```

## El gate de finalist

| condicion | valor | se cumple |
|---|---|---|
| delta >= +0,003 | +0.0026743167 | False |
| pliegues >= 4/5 | 4/5 | True |
| bootstrap favorable | incluye cero: True | False |
| no lo explica el barajado | +0.0041133588 | True |
| rescate de alta confianza | 6 | True |

## Veredicto

```text
Cumple 3 de las cinco condiciones del gate de finalist: delta +0.0026743167, 4 de 5 pliegues, IC95 [-0.0044311634, +0.0102108745] incluye cero: True, brecha +0.0041133588, rescate de alta confianza 6. Es el mejor resultado de toda la investigacion en el espacio de scores y queda como PROMISING

DECISION: PROMISING
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p66_03_complementariedad_bootstrap.py`. Sin push.*