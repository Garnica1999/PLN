# P60-L - Riesgo minimax por grupos (DRO)

Generado por `experiments/p60_03_dro.py`. Fecha 2026-09-28T17:53:45+00:00.

## Motivacion e hipotesis

V03 optimiza un unico riesgo promedio. Si ese promedio esconde comportamientos
sistematicamente distintos en subpoblaciones, un objetivo minimax sobre unos pocos
**grupos que ya existen** deberia colocar la frontera mejor que el promedio.

## Por que los grupos son los cuartiles de longitud

Los grupos near-duplicate y pseudo-autor no sirven aqui por una razon mecanica: son el
propio grupo del CV, con 1415 grupos para 1483 ejemplos, luego un minimax sobre ellos
degenera en un objetivo de peor ejemplo, que no es DRO de grupos.

El cuartil de longitud tiene evidencia. `V28-01` midio que la longitud es el rasgo que
mas discrimina, con Cohen d de +0,28 a +0,34 a favor de los textos largos, y que es el
mismo para las tres etiquetas. La definicion es la del proyecto, `pd.qcut` sobre
`n_caracteres`, usada en `06_error_analysis.py` y `08_features_densas_h11_h12.py`.

## Heterogeneidad medida en este corpus

| cuartil | n | caracteres | F1 media | error |
|---|---:|---|---|---:|
| Q1 | 390 | 10-60 | 0.8895 | 11.54 por ciento |
| Q2 | 361 | 61-92 | 0.7638 | 20.13 por ciento |
| Q3 | 361 | 93-131 | 0.7204 | 16.53 por ciento |
| Q4 | 371 | 132-306 | 0.5980 | 20.22 por ciento |

## Configuracion, fijada antes de mirar validation

```text
grupos   los cuatro cuartiles de longitud, definidos SOLO con training del pliegue
objetivo DRO = (1/1.0) log sum_g (n_g/n) exp(1.0 riesgo_g)
eta      1.0, el valor del trabajo con gradiente exponenciado, un solo valor
pesos    los mismos balanceados de V03   |   L2  C = 2,0   |   umbrales  congelados
rejilla  ninguna sobre eta ni sobre el numero de grupos
```

## Verificacion del gradiente, antes de entrenar

| configuracion | error relativo w | error relativo b |
|---|---:|---:|
| un_grupo | 6.73e-07 | 1.16e-10 |
| cuatro_grupos | 1.42e-06 | 6.52e-09 |

## Control

El brazo 0 mete todas las filas de entrenamiento en UN solo grupo, lo que reduce DRO al
riesgo promedio global, que es el objetivo de V03. Por eso el brazo 0 es un control real.

```text
brazo 0 corrected_cv 0.7772647318762402  delta vs V03 +0.0000532794  celdas distintas 3
```

## Resultados

```text
brazo 0 (un grupo)   corrected_cv 0.7772647319
brazo 1 (cuatro)     corrected_cv 0.7776009860
delta DRO PURO       +0.0003362541
delta vs V03         +0.0003895335
pliegues que mejoran 2/5
rescate 2 | dano 1 | neto +1 | cambian 3
convergencia True
```

| etiqueta | F1 DRO | F1 control |
|---|---:|---:|
| HS | 0.7617 | 0.7623 |
| TR | 0.8176 | 0.8153 |
| AG | 0.7535 | 0.7542 |

## Por grupo, que es lo que DRO promete equitablear

| cuartil | n | control | DRO | delta |
|---|---:|---:|---:|---:|
| Q1 | 390 | 0.8902 | 1.0000 | +0.0001 |
| Q2 | 361 | 0.7648 | 1.0000 | +0.0000 |
| Q3 | 361 | 0.7176 | 1.0000 | +0.0027 |
| Q4 | 371 | 0.5980 | 1.0000 | -0.0009 |

## Gate

| criterio | valor | superado |
|---|---|---|
| efecto de la agrupacion >= +0,003 | +0.000336 | **False** |
| al menos 4 de 5 pliegues | 2/5 | **False** |
| gate completo | | **False** |

Bootstrap NO ejecutado: el gate no se supero.

## Decision

```text
decision  MARGINAL. El mecanismo se ACTIVA pero con +2,5 por ciento de sobreponderacion del grupo peor, solo cambian 3 de 4449 celdas y el grupo que peor lo tenia empeora; ademas no alcanza el gate de +0,003
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_03_dro.py`. Sin push.*