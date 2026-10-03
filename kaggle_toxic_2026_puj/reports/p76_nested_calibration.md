# P76: calibracion nested de V03 y del TextCNN

`experiment_id` P76 | inicio 2026-09-30T09:48:39+00:00 | solo raw y Platt | alphas [0.05, 0.1]

## La hipotesis

> La TextCNN contiene informacion complementaria respecto de V03, pero sus probabilidades crudas no estan calibradas de forma compatible con V03. La calibracion nested de ambos modelos podria reducir el damage de la fusion y permitir que parte del rescue se transfiera a `corrected_cv`.

## El protocolo nested

```text
outer_train -> inner_train + inner_calibration -> calibrador -> outer_validation
la outer_validation permanece COMPLETAMENTE aislada
```

El calibrador se ajusta sobre `inner_calibration` con modelos entrenados en `inner_train` (unas 830 filas), y se aplica a las predicciones de P75, que vienen de modelos entrenados sobre el `outer_train` completo (1186). **Esa asimetria se declara.**

## Calibracion: metricas sobre la outer_validation

| modelo | macro Brier | macro log loss | pendiente HS | pendiente TR | pendiente AG |
|---|---|---|---|---|---|
| v03_raw | 0.125771 | 0.403583 | 1.311048 | 1.360647 | 1.298334 |
| v03_cal | 0.122455 | 0.391009 | 0.996569 | 0.998942 | 0.974155 |
| cnn_raw | 0.192150 | 0.576984 | 0.59205 | 0.61502 | 0.589011 |
| cnn_cal | 0.188485 | 0.555991 | 0.813262 | 0.861862 | 0.89355 |

## Los cuatro miembros con los umbrales congelados

```text
V03_raw  0.7772114525   delta vs campeon +0.0000000000
V03_cal  0.7685825353   delta vs campeon -0.0086289172
CNN_raw  0.5838755350   delta vs campeon -0.1933359176
CNN_cal  0.5738807461   delta vs campeon -0.2033307064
```

## Las ocho configuraciones de fusion

```text
A_V03raw_CNNraw_alpha0.05      cv 0.7746109462  delta -0.0026005064  folds 2/5  rescue   17  damage   21  net    -4
A_V03raw_CNNraw_alpha0.10      cv 0.7719675587  delta -0.0052438938  folds 1/5  rescue   31  damage   45  net   -14
B_V03cal_CNNraw_alpha0.05      cv 0.7651264389  delta -0.0120850136  folds 1/5  rescue   68  damage   72  net    -4
B_V03cal_CNNraw_alpha0.10      cv 0.7624993298  delta -0.0147121227  folds 1/5  rescue   72  damage   81  net    -9
C_V03raw_CNNcal_alpha0.05      cv 0.7754370605  delta -0.0017743921  folds 2/5  rescue   11  damage   14  net    -3
C_V03raw_CNNcal_alpha0.10      cv 0.7749963696  delta -0.0022150829  folds 2/5  rescue   24  damage   27  net    -3
D_V03cal_CNNcal_alpha0.05      cv 0.7659669038  delta -0.0112445488  folds 1/5  rescue   67  damage   69  net    -2
D_V03cal_CNNcal_alpha0.10      cv 0.7645497894  delta -0.0126616631  folds 1/5  rescue   71  damage   75  net    -4
```

## Diversidad

```text
CNN_raw  AUC sobre el error de V03 {'HS': 0.487206, 'TR': 0.604502, 'AG': 0.497264}  acuerdo 0.757024
CNN_cal  AUC sobre el error de V03 {'HS': 0.491008, 'TR': 0.603231, 'AG': 0.499461}  acuerdo 0.748483
la calibracion del CNN cambia 0.051697 de sus decisiones
```

## Veredicto

**CASO B: CALIBRACION_MEJORA_COMPLEMENTARIEDAD_PERO_NO_F1**

Finalista: **None**

## Limitaciones

```text
- el calibrador se ajusta sobre un modelo mas debil que aquel al que se aplica
- el bootstrap solo se ejecuta si el gate preliminar lo exige, y ninguna configuracion lo supero
- los tiempos de las corridas anteriores estan contaminados por carga externa y no se usan para comparaciones
```
