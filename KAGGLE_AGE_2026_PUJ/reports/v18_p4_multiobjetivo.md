# V18 · P4 — Entrenamiento multi-objetivo

## 1. Diseno

P1 y P3 dejaron dos orientaciones: un objetivo que **redistribuye** no puede ganar en una
media de seis clases, y las dos fronteras estan **acopladas** por la clase `25-34`, luego no
se pueden optimizar por separado. De ahi un unico termino auxiliar **aditivo** sobre las dos
fronteras a la vez, y tres brazos que separan arquitectura de objetivo:

| Brazo | Que cambia | Parametros extra |
|---|---|---|
| A, control | nada | 0 |
| B, arquitectura | dos cabezas binarias que reparten la masa que la global ya pone en cada par | 558 |
| C, objetivo | un termino binario de frontera sobre los **mismos** logit | **0** |

Todo pre-registrado: 30 epocas fijas sin early stopping, lote 512, Adam 0.001, lambda en {0,1, 0,5} y margen fijo. La representacion son las mismas **279 variables de V12** en los tres brazos, sin ninguna nueva.

## 2. Resultado de la cabeza global, y del sistema combinado con el MLP de V12

| Brazo | Lambda | Param. extra | macro global | frontera 1 | frontera 2 | macro combinada | Delta | IC95 |
|---|---|---|---|---|---|---|---|---|
| A_control | 0.0 | 0 | 0.5874 | 0.4858 | 0.4829 | 0.5982 | -0.0023 | [-0.0031, -0.0015] |
| B_arquitectura | 0.0 | 558 | 0.5893 | 0.4842 | 0.4868 | 0.5993 | -0.0012 | [-0.0019, -0.0004] |
| C1_objetivo | 0.1 | 0 | 0.5879 | 0.4847 | 0.4808 | 0.599 | -0.0015 | [-0.0023, -0.0008] |
| C2_objetivo | 0.5 | 0 | 0.5887 | 0.4896 | 0.4889 | 0.599 | -0.0015 | [-0.0023, -0.0008] |
| V12 (referencia) | - | - | - | 0.4963 | 0.4974 | 0.6005 | - | - |

## 3. Salud del entrenamiento

| Brazo | F1 de train | F1 de validacion | Brecha | Epocas |
|---|---|---|---|---|
| A_control | 0.5916 | 0.5874 | 0.0042 | 30 |
| B_arquitectura | 0.5919 | 0.5893 | 0.0026 | 30 |
| C1_objetivo | 0.5927 | 0.5879 | 0.0048 | 30 |
| C2_objetivo | 0.5927 | 0.5887 | 0.004 | 30 |

## 4. F1 por clase del sistema combinado

| Brazo | 13-17 | 18-24 | 25-34 | 35-49 | 50-64 | 65-xx |
|---|---|---|---|---|---|---|
| A_control | 0.6667 | 0.5042 | 0.4845 | 0.5561 | 0.6549 | 0.723 |
| B_arquitectura | 0.6684 | 0.505 | 0.4867 | 0.5577 | 0.6544 | 0.7239 |
| C1_objetivo | 0.6689 | 0.5065 | 0.4833 | 0.5581 | 0.6535 | 0.7235 |
| C2_objetivo | 0.6683 | 0.5044 | 0.485 | 0.5575 | 0.6549 | 0.7238 |
| V12 (referencia) | 0.671 | 0.5059 | 0.4858 | 0.5593 | 0.6549 | 0.7261 |
