# P56 - Transferencia de umbrales corregida

Generado por `experiments/v56_00_transferencia_umbrales.py` (V56-00 (transferencia de umbrales con anidamiento completo)).

Fecha 2026-09-28T03:01:13+00:00. **0 submissions, 0 consultas a Kaggle, 0 uso de holdout o test. La unica variable es el procedimiento de estimacion de umbrales.**

## 1. Reproduccion de P55

- valor registrado: **0.7781367700**
- valor reproducido: **0.7781367700**
- `P55_CORRECTED_RESULT_REPRODUCED = True`

## 2. El procedimiento, que aqui SI es un anidamiento completo

- outer: 5 pliegues canonicos de TOXIC, grupos near-dup UNION pseudo-autor
- inner: 4 pliegues agrupados dentro de cada outer train, con vectorizador y LogisticRegression re-ajustados
- rejilla: 0,20 a 0,80 paso 0,01, la misma rejilla historica
- agregacion: mediana de las cinco estimaciones. Una sola regla
- outer_validation_participa: **en nada del ajuste**

La diferencia con P55 es que aqui los umbrales se ajustan sobre un OOF **interior**, con vectorizador y modelo re-ajustados, en vez de sobre el OOF de los cuatro pliegues de entrenamiento.

## 3. Umbrales por pliegue exterior y distribucion

| outer | inner HS | inner TR | inner AG | V03 HS | V03 TR | V03 AG |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.40 | 0.47 | 0.40 | 0.44 | 0.51 | 0.46 |
| 1 | 0.41 | 0.46 | 0.48 | 0.44 | 0.51 | 0.46 |
| 2 | 0.42 | 0.46 | 0.43 | 0.44 | 0.51 | 0.46 |
| 3 | 0.41 | 0.51 | 0.51 | 0.44 | 0.51 | 0.46 |
| 4 | 0.45 | 0.46 | 0.45 | 0.44 | 0.51 | 0.46 |

| etiqueta | estimaciones | media | mediana | std | min | max | V03 | V03 en el rango | V03 - mediana |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| HS | 0.40 0.41 0.42 0.41 0.45 | 0.4180 | **0.41** | 0.0192 | 0.40 | 0.45 | 0.44 | True | -0.03 |
| TR | 0.47 0.46 0.46 0.51 0.46 | 0.4720 | **0.46** | 0.0217 | 0.46 | 0.51 | 0.51 | True | -0.05 |
| AG | 0.40 0.48 0.43 0.51 0.45 | 0.4540 | **0.45** | 0.0428 | 0.40 | 0.51 | 0.46 | True | -0.01 |

## 4. T0 contra T1

| politica | macro-F1 | delta vs V03 |
|---|---:|---:|
| T0 historico (0,44/0,51/0,46) | 0.7772114525 | +0.0000000000 |
| **T1 mediana de las cinco** | **0.7798514633** | **+0.0026400107** |
| umbral por pliegue exterior (plenamente honesto) | 0.7745478730 | -0.0026635795 |
| P55 simple, para referencia | 0.7781367700 | +0.0009253174 |

### F1 por etiqueta

| etiqueta | T0 | T1 | delta |
|---|---:|---:|---:|
| HS | 0.7616892911 | 0.7675597393 | +0.0058704482 |
| TR | 0.8157303371 | 0.8156182213 | -0.0001121158 |
| AG | 0.7542147294 | 0.7563764292 | +0.0021616998 |

### Tasa positiva

| etiqueta | T0 | T1 | diferencia |
|---|---:|---:|---:|
| HS | 0.4572 | 0.4943 | +0.0371 |
| TR | 0.3001 | 0.3216 | +0.0216 |
| AG | 0.3776 | 0.3844 | +0.0067 |

### Por pliegue

| pliegue | macro-F1 de T1 | delta vs V03 global |
|---:|---:|---:|
| 0 | 0.7964112611 | +0.0191998086 |
| 1 | 0.7941841599 | +0.0169727074 |
| 2 | 0.7824619463 | +0.0052504938 |
| 3 | 0.7461905211 | -0.0310209314 |
| 4 | 0.7808003406 | +0.0035888881 |

Pliegues con delta positivo: **4 de 5**.

## 5. Complementariedad, T1 frente a T0

| categoria | celdas |
|---|---:|
| ambos correctos | 3638 |
| ambos incorrectos | 714 |
| T0 mal, T1 bien (rescate) | 43 |
| T0 bien, T1 mal (damage) | 54 |
| **net** | **-11** |

| etiqueta | rescate | damage | net |
|---|---:|---:|---:|
| HS | 25 | 30 | -5 |
| TR | 13 | 19 | -6 |
| AG | 5 | 5 | +0 |

## 6. Bootstrap

| magnitud | valor |
|---|---:|
| mediana del delta | +0.002553 |
| IC95 | [-0.003456, +0.008989] |
| incluye 0 | True |
| replicas | 2000 |

## 7. Decision

- umbrales de V03 fuera del rango interior en las cinco estimaciones: **[]**
- delta de T1: **+0.002640**, con **4 de 5** pliegues positivos
- `THRESHOLD_TRANSFER_BIAS = NOT_CONFIRMED`
- `STATUS = CLOSED`
- `CASE = B`

