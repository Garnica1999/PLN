# P61 - Capacidad LightGBM con Optuna anidado

Generado por `experiments/p61_03_lightgbm.py`. Fecha 2026-09-28T20:26:23+00:00.

## El gap check que autoriza la familia, y el riesgo que expone

LightGBM, XGBoost y CatBoost devuelven **0 filas** en el registro canonico. La familia
es genuinamente nueva, que es lo que pide el encargo para gastar experimentos en ella.

Pero el registro tambien contiene el precedente que mas pesa, y no es tranquilizador.
`OPT-01` ejecuto Optuna sobre el propio TF-IDF de V03, 24 trials, y gano **+0,008565**
en la puntuacion out-of-fold. Se descarto porque **no transfirio**: 0,77602 frente a
0,78306 de V03 en el leaderboard, un hueco de -0,028. Su conclusion registrada es que
**la busqueda por OOF premia capacidad que no generaliza al test**.

La situacion es: una familia autorizada y nunca probada, en un proyecto donde una busqueda
de capacidad ya probada produjo un falso positivo que solo el leaderboard pudo detectar.
Por eso el gate se aplica a la puntuacion **anidada** y no a la mejor puntuacion de trial.

## Representacion, y por que es la que ya tiene V03

El encargo lista el TF-IDF de V03 primero y el SVD como alternativa densa. El SVD no se
usa, y el motivo esta en el registro: `V4-G01-G03` ya ajusto gradient boosting sobre SVD
de este corpus y dio `svd:hist_gb 0,7297` y `svd:extra_trees 0,7033`, muy por debajo de
V03. Importar esa representacion ya sabidamente debil mezclaria el cambio de familia con
un cambio de representacion, que es justo lo que el parrafo 23 prohibe.

El regimen se declara como riesgo: **27.431 rasgos frente a 1.186 filas** de
entrenamiento es un regimen donde los arboles no son naturalmente fuertes.

## Los tres brazos, que el parrafo 27 exige

| brazo | corrected_cv | delta vs V03 |
|---|---:|---:|
| 0, V03 referencia | 0.7772114525 | +0.0000000000 |
| 1, LightGBM simple | 0.7469717334 | -0.0302397191 |
| 2, LightGBM + Optuna | 0.7513651502 | -0.0258463023 |

```text
cambio de FAMILIA     (brazo 1 - 0) -0.0302397191
optimizacion de params(brazo 2 - 1) +0.0043934168
total                 (brazo 2 - 0) -0.0258463023
```

Reportar solo uno de los dos primeros es el error que el parrafo 27 esta escrito para
evitar, porque son mecanismos distintos: uno cambia de familia de modelo y el otro cambia
los hiperparametros de esa familia.

## El anidamiento, y el objetivo de la busqueda

```text
outer_train -> inner_train, inner_validation, partidos POR GRUPO
           -> Optuna, que solo ve la validacion interna
           -> refit sobre el outer_train completo con los parametros ganadores
outer_validation, que no participa en nada
```

El objetivo de Optuna es el **log loss de la validacion interna**, no el F1 interno. Es
deliberado: usar F1 dentro de la busqueda haria depender la busqueda del umbral de
decision, y en esta fase los umbrales estan congelados. Una regla de puntuacion
apropiada es libre de umbral, luego la busqueda optimiza el modelo de probabilidad y
los umbrales de V03 se aplican una sola vez, al final, sobre la validacion exterior.

## Resultados

| etiqueta | F1 brazo 1 | F1 brazo 2 | F1 V03 |
|---|---:|---:|---:|
| HS | 0.7319 | 0.7340 | 0.7617 |
| TR | 0.7765 | 0.7779 | 0.8157 |
| AG | 0.7325 | 0.7422 | 0.7542 |

| pliegue | inner_train | inner_val | mejor logloss interno | mejor iteracion |
|---:|---:|---:|---:|---:|
| 1 | 951 | 235 | 0.288150 | 535 |
| 2 | 945 | 241 | 0.259054 | 1210 |
| 3 | 951 | 236 | 0.361307 | 76 |
| 4 | 947 | 240 | 0.342003 | 185 |
| 5 | 949 | 237 | 0.313872 | 774 |

```text
pliegues donde el candidato 2 mejora sobre V03   2/5
rescate 225 | dano 253 | neto -28 | cambian 478
gate superado False | bootstrap NO EJECUTADO
trials totales 450
```

## Reproducibilidad

Nivel **R2**: semilla fija, sampler y semilla registrados, artefacto en disco con los
450 trials y hash del OOF verificado. Una reejecucion con la misma
semilla dara el mismo resultado, pero no se ha reproducido de forma independiente.

## Decision

```text
decision  CLOSED_BY_DUPLICATION. La familia no aporta sobre V03 y el precedente OPT-01 ya cerro el patron de busqueda de capacidad por OOF
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p61_03_lightgbm.py`. Sin push.*