# P61 - Segunda particion sobre los finalistas congelados

Generado por `experiments/p61_08_segunda_particion.py`. Fecha 2026-09-28T21:05:39+00:00.

## Que es finalista y que no

```text
finalistas congelados: con_nuevo_k3_A, con_nuevo_k4_A
excluido: con_nuevo_k5_A, supera el delta y los 5/5 pliegues, pero su IC95 de bootstrap es [-0,001798, +0,008766], que CONTIENE el cero, luego no cumple la condicion de bootstrap favorable y no es finalista
```

El tercero superaba el delta y los 5/5 pliegues, pero su intervalo de bootstrap
contiene el cero, luego no cumple la condicion de bootstrap favorable. Se excluye y se
dice por que, en vez de dejarlo en silencio.

## Que se revalida, y que no

se revalida la CAPA DE DECISION, es decir la eleccion de pertenencia y la de pesos. NO se revalidan los modelos base, porque los veinte canales historicos solo existen como probabilidades out-of-fold en la particion canonica. Reajustarlos exigiria su codigo de entrenamiento exacto y sus hiperparametros, y adivinarlos sustituiria en silencio un modelo distinto por uno con nombre

## Las dos particiones

```text
canonica  hash 81bc0a3102bf1bc7b9a5e142d50072d6...
nueva     hash 4f0ea95046da0a15be1885f180f20acd...  semilla 20260928
grupos canonicos partidos entre pliegues  0
```

## Test 1, particion nueva

| configuracion | corrected_cv | delta | pliegues + |
|---|---:|---:|---:|
| test1_con_nuevo_A | 0.7760586556 | -0.0011527969 | 2/5 |
| test1_con_nuevo_B | 0.7788843359 | +0.0016728834 | 3/5 |
| test1_sin_nuevo_A | 0.7816740411 | +0.0044625886 | 4/5 |
| test1_sin_nuevo_B | 0.7787776014 | +0.0015661489 | 2/5 |

## Test 2, holdback tres contra dos sobre las diez parejas

| internas | delta en las externas |
|---|---:|
| pliegues 1v2 | +0.005130 |
| pliegues 1v3 | +0.000453 |
| pliegues 1v4 | +0.006719 |
| pliegues 1v5 | +0.003046 |
| pliegues 2v3 | +0.003121 |
| pliegues 2v4 | +0.001686 |
| pliegues 2v5 | +0.007309 |
| pliegues 3v4 | -0.011350 |
| pliegues 3v5 | -0.018220 |
| pliegues 4v5 | -0.013802 |

```text
delta medio -0.001591 | mediana +0.002366
positivas 7/10 | minimo -0.018220 | maximo +0.007309
```

## Test 3, control negativo

Los miembros historicos se sustituyen por los PEORES canales del pool, que estan entre
0,04 y 0,24 por debajo de V03:

```text
  cita:NEG                   corrected_cv 0.533063
  synt:SYN-MORPH             corrected_cv 0.565239
  synt:SYN-DEPS              corrected_cv 0.572844

delta con esos miembros: +0.0059889238
```

Este es el test que decide si la explicacion es correcta. Los otros dos solo preguntan
si el numero es reproducible; este pregunta si la causa es la que se afirma.

## Bootstrap sobre la particion nueva

```text
IC95 agrupado por los 1415 grupos canonicos  [-0.011307, +0.031095]
incluye el cero  True
```

## Veredicto

```text
el control negativo NO colapsa: la ganancia aparece igual con miembros que estan 0,04 a 0,24 por debajo de V03, luego la explicacion por diversidad no se sostiene y el efecto hay que atribuirlo al hecho de ejecutar una busqueda de pesos sobre un pool que incluye V03
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p61_08_segunda_particion.py`. Sin push.*